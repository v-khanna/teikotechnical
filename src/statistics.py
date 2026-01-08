"""
Statistical analysis module for clinical trial data.

Implements Part 3: Compare responders vs non-responders for melanoma patients
receiving miraclib treatment (PBMC samples only).
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .database import get_db_path, get_connection, get_project_root, CELL_POPULATIONS


def get_response_comparison_data(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Get filtered data for response comparison analysis.
    
    Filters: melanoma + miraclib + PBMC (all timepoints)
    
    Args:
        conn: SQLite connection. Creates new connection if not provided.
        
    Returns:
        DataFrame with sample metadata and cell percentages.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    
    # Query filtered data with cell counts
    query = """
        SELECT 
            s.sample_id as sample,
            s.subject,
            s.response,
            s.time_from_treatment_start,
            cc.population,
            cc.count
        FROM samples s
        JOIN cell_counts cc ON s.sample_id = cc.sample_id
        WHERE s.condition = 'melanoma'
          AND s.treatment = 'miraclib'
          AND s.sample_type = 'PBMC'
          AND s.response IN ('yes', 'no')
        ORDER BY s.sample_id, cc.population
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Calculate total counts and percentages per sample
    totals = df.groupby('sample')['count'].sum().reset_index()
    totals.columns = ['sample', 'total_count']
    
    df = df.merge(totals, on='sample')
    df['percentage'] = (df['count'] / df['total_count']) * 100
    
    if close_conn:
        conn.close()
    
    return df


def perform_mann_whitney_test(responders: np.ndarray, non_responders: np.ndarray) -> Dict:
    """
    Perform Mann-Whitney U test between two groups.
    
    Args:
        responders: Array of values for responders
        non_responders: Array of values for non-responders
        
    Returns:
        Dictionary with test results.
    """
    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(
        responders, 
        non_responders, 
        alternative='two-sided'
    )
    
    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(responders), len(non_responders)
    effect_size = 1 - (2 * statistic) / (n1 * n2)
    
    return {
        'test_used': 'Mann-Whitney U',
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'responder_n': n1,
        'non_responder_n': n2
    }


def analyze_population_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze differences between responders and non-responders for each cell population.
    
    Args:
        df: DataFrame from get_response_comparison_data()
        
    Returns:
        DataFrame with statistical results for each population.
    """
    results = []
    
    for population in CELL_POPULATIONS:
        pop_data = df[df['population'] == population]
        
        responders = pop_data[pop_data['response'] == 'yes']['percentage'].values
        non_responders = pop_data[pop_data['response'] == 'no']['percentage'].values
        
        # Calculate descriptive statistics
        resp_mean = np.mean(responders)
        resp_std = np.std(responders)
        resp_median = np.median(responders)
        
        non_resp_mean = np.mean(non_responders)
        non_resp_std = np.std(non_responders)
        non_resp_median = np.median(non_responders)
        
        # Perform statistical test
        test_results = perform_mann_whitney_test(responders, non_responders)
        
        results.append({
            'population': population,
            'responder_mean': round(resp_mean, 2),
            'responder_std': round(resp_std, 2),
            'responder_median': round(resp_median, 2),
            'non_responder_mean': round(non_resp_mean, 2),
            'non_responder_std': round(non_resp_std, 2),
            'non_responder_median': round(non_resp_median, 2),
            'responder_n': test_results['responder_n'],
            'non_responder_n': test_results['non_responder_n'],
            'test_used': test_results['test_used'],
            'statistic': round(test_results['statistic'], 2),
            'p_value': test_results['p_value'],
            'effect_size': round(test_results['effect_size'], 4)
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    n_tests = len(CELL_POPULATIONS)
    results_df['adjusted_p_value'] = results_df['p_value'] * n_tests
    results_df['adjusted_p_value'] = results_df['adjusted_p_value'].clip(upper=1.0)
    
    # Determine significance (alpha = 0.05 after Bonferroni correction = 0.01 per test)
    results_df['significant'] = results_df['adjusted_p_value'] < 0.05
    results_df['significant'] = results_df['significant'].map({True: 'Yes', False: 'No'})
    
    # Format p-values for display
    results_df['p_value_formatted'] = results_df['p_value'].apply(
        lambda x: f'{x:.2e}' if x < 0.001 else f'{x:.4f}'
    )
    results_df['adjusted_p_formatted'] = results_df['adjusted_p_value'].apply(
        lambda x: f'{x:.2e}' if x < 0.001 else f'{x:.4f}'
    )
    
    return results_df


def create_boxplots(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Create boxplots comparing responders vs non-responders for each cell population.
    
    Args:
        df: DataFrame from get_response_comparison_data()
        output_path: Path for output file. Uses default if not provided.
        
    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = get_project_root() / 'outputs' / 'part3_boxplots.png'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    # Color palette
    colors = {'yes': '#2ecc71', 'no': '#e74c3c'}
    
    # Create a boxplot for each population
    for idx, population in enumerate(CELL_POPULATIONS):
        ax = axes[idx]
        pop_data = df[df['population'] == population]
        
        # Create boxplot
        sns.boxplot(
            data=pop_data,
            x='response',
            y='percentage',
            hue='response',
            order=['yes', 'no'],
            hue_order=['yes', 'no'],
            palette=colors,
            ax=ax,
            width=0.6,
            legend=False
        )
        
        # Customize appearance
        ax.set_xlabel('')
        ax.set_ylabel('Relative Frequency (%)', fontsize=10)
        ax.set_title(f'{population.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Responders', 'Non-responders'], fontsize=10)
        
        # Add sample size annotations
        n_resp = len(pop_data[pop_data['response'] == 'yes'])
        n_non = len(pop_data[pop_data['response'] == 'no'])
        ax.text(0, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                f'n={n_resp}', ha='center', fontsize=9, color='#666666')
        ax.text(1, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                f'n={n_non}', ha='center', fontsize=9, color='#666666')
    
    # Hide the 6th subplot (we only have 5 populations)
    axes[5].set_visible(False)
    
    # Add overall title
    fig.suptitle('Cell Population Frequencies: Responders vs Non-responders\n'
                 '(Melanoma patients treated with Miraclib, PBMC samples)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Boxplots saved to {output_path}")
    
    return output_path


def save_statistics(results_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save statistical results to CSV file.
    
    Args:
        results_df: DataFrame from analyze_population_differences()
        output_path: Path for output file. Uses default if not provided.
        
    Returns:
        Path to saved file.
    """
    if output_path is None:
        output_path = get_project_root() / 'outputs' / 'part3_statistics.csv'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select and order columns for output
    output_cols = [
        'population',
        'responder_mean', 'responder_std', 'responder_median', 'responder_n',
        'non_responder_mean', 'non_responder_std', 'non_responder_median', 'non_responder_n',
        'test_used', 'statistic', 'p_value', 'adjusted_p_value', 'effect_size', 'significant'
    ]
    
    results_df[output_cols].to_csv(output_path, index=False)
    print(f"Statistical results saved to {output_path}")
    
    return output_path


def print_findings_summary(results_df: pd.DataFrame) -> None:
    """
    Print a summary of the statistical findings.
    
    Args:
        results_df: DataFrame from analyze_population_differences()
    """
    print("\n" + "=" * 60)
    print("STATISTICAL FINDINGS SUMMARY")
    print("=" * 60)
    
    significant_pops = results_df[results_df['significant'] == 'Yes']['population'].tolist()
    
    if significant_pops:
        print(f"\nSignificant differences found in {len(significant_pops)} population(s):")
        for pop in significant_pops:
            row = results_df[results_df['population'] == pop].iloc[0]
            direction = "higher" if row['responder_mean'] > row['non_responder_mean'] else "lower"
            print(f"  - {pop}: Responders have {direction} frequencies")
            print(f"    (Responder: {row['responder_mean']:.2f}% vs Non-responder: {row['non_responder_mean']:.2f}%)")
            print(f"    (Adjusted p-value: {row['adjusted_p_formatted']}, Effect size: {row['effect_size']:.4f})")
    else:
        print("\nNo statistically significant differences found after Bonferroni correction.")
    
    print("\n" + "-" * 60)
    print("Note: Statistical significance determined using Mann-Whitney U test")
    print(f"      with Bonferroni correction (alpha = 0.05/{len(CELL_POPULATIONS)} = 0.01 per test)")
    print("=" * 60)


def run_part3(conn: Optional[sqlite3.Connection] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Part 3 analysis: Statistical comparison and visualization.
    
    Args:
        conn: SQLite connection. Creates new connection if not provided.
        
    Returns:
        Tuple of (comparison_data, statistics_results) DataFrames.
    """
    print("\n" + "=" * 60)
    print("PART 3: Statistical Analysis - Response Comparison")
    print("=" * 60)
    
    # Get filtered data
    print("\nFiltering data: melanoma + miraclib + PBMC...")
    df = get_response_comparison_data(conn)
    
    n_samples = df['sample'].nunique()
    n_responders = df[df['response'] == 'yes']['sample'].nunique()
    n_non_responders = df[df['response'] == 'no']['sample'].nunique()
    
    print(f"  - Total samples: {n_samples}")
    print(f"  - Responders: {n_responders} samples")
    print(f"  - Non-responders: {n_non_responders} samples")
    
    # Perform statistical analysis
    print("\nPerforming statistical tests...")
    results_df = analyze_population_differences(df)
    
    # Print results table
    print("\nStatistical Results:")
    print("-" * 60)
    display_cols = ['population', 'responder_mean', 'non_responder_mean', 
                    'p_value_formatted', 'adjusted_p_formatted', 'significant']
    print(results_df[display_cols].to_string(index=False))
    
    # Save outputs
    print("\nSaving outputs...")
    create_boxplots(df)
    save_statistics(results_df)
    
    # Print findings summary
    print_findings_summary(results_df)
    
    return df, results_df


if __name__ == "__main__":
    # Run analysis when executed directly
    from .database import get_connection
    
    conn = get_connection()
    run_part3(conn)
    conn.close()
    print("\nPart 3 analysis complete.")

