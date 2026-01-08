"""
Analysis module for clinical trial data.

Implements Part 2 (cell frequency calculations) and Part 4 (baseline subset analysis).
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from .database import get_db_path, get_connection, get_project_root


def calculate_cell_frequencies(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Calculate relative frequencies of each cell population for all samples.
    
    Part 2: Transform data to long format with percentage calculations.
    
    For each sample:
    - total_count = sum of all 5 cell population counts
    - percentage = (count / total_count) * 100
    
    Args:
        conn: SQLite connection. Creates new connection if not provided.
        
    Returns:
        DataFrame with columns: sample, total_count, population, count, percentage
        Contains 52,500 rows (10,500 samples × 5 populations)
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    
    # Query all cell counts with sample information
    query = """
        SELECT 
            cc.sample_id as sample,
            cc.population,
            cc.count
        FROM cell_counts cc
        ORDER BY cc.sample_id, cc.population
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Calculate total count per sample
    totals = df.groupby('sample')['count'].sum().reset_index()
    totals.columns = ['sample', 'total_count']
    
    # Merge totals back to main dataframe
    df = df.merge(totals, on='sample')
    
    # Calculate percentage
    df['percentage'] = (df['count'] / df['total_count']) * 100
    
    # Round percentage to 2 decimal places for readability
    df['percentage'] = df['percentage'].round(2)
    
    # Reorder columns to match specification
    df = df[['sample', 'total_count', 'population', 'count', 'percentage']]
    
    # Sort by sample and population for consistent output
    df = df.sort_values(['sample', 'population']).reset_index(drop=True)
    
    if close_conn:
        conn.close()
    
    return df


def validate_frequencies(df: pd.DataFrame) -> bool:
    """
    Validate that percentages sum to 100% for each sample.
    
    Args:
        df: DataFrame from calculate_cell_frequencies()
        
    Returns:
        True if validation passes, False otherwise.
    """
    grouped = df.groupby('sample')['percentage'].sum()
    
    # Check if all sums are approximately 100 (allowing for rounding)
    tolerance = 0.1  # Allow 0.1% tolerance for rounding errors
    valid = (grouped >= 100 - tolerance).all() and (grouped <= 100 + tolerance).all()
    
    if not valid:
        print("WARNING: Some samples don't sum to 100%:")
        invalid = grouped[(grouped < 100 - tolerance) | (grouped > 100 + tolerance)]
        print(invalid.head(10))
    
    return valid


def save_summary_table(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save the cell frequency summary table to CSV.
    
    Args:
        df: DataFrame from calculate_cell_frequencies()
        output_path: Path for output file. Uses default if not provided.
        
    Returns:
        Path to saved file.
    """
    if output_path is None:
        output_path = get_project_root() / 'outputs' / 'part2_summary_table.csv'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Summary table saved to {output_path}")
    print(f"  - Total rows: {len(df)}")
    
    return output_path


def analyze_baseline_subset(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """
    Analyze the baseline subset for Part 4.
    
    Filters: melanoma + PBMC + baseline (day 0) + miraclib treatment
    
    Reports:
    - Total samples matching criteria
    - Total unique subjects
    - Samples per project
    - Responders vs non-responders count
    - Males vs females count
    
    Args:
        conn: SQLite connection. Creates new connection if not provided.
        
    Returns:
        Dictionary with all analysis results.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    
    cursor = conn.cursor()
    
    results = {}
    
    # Base filter for Part 4
    base_filter = """
        condition = 'melanoma' 
        AND sample_type = 'PBMC' 
        AND time_from_treatment_start = 0 
        AND treatment = 'miraclib'
    """
    
    # Total samples matching criteria
    cursor.execute(f"""
        SELECT COUNT(*) FROM samples WHERE {base_filter}
    """)
    results['total_samples'] = cursor.fetchone()[0]
    
    # Total unique subjects
    cursor.execute(f"""
        SELECT COUNT(DISTINCT subject) FROM samples WHERE {base_filter}
    """)
    results['unique_subjects'] = cursor.fetchone()[0]
    
    # Samples per project
    cursor.execute(f"""
        SELECT project, COUNT(*) as count 
        FROM samples 
        WHERE {base_filter}
        GROUP BY project
        ORDER BY project
    """)
    results['samples_per_project'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Check for projects with no data (prj2 should have 0)
    cursor.execute("SELECT DISTINCT project FROM samples ORDER BY project")
    all_projects = [row[0] for row in cursor.fetchall()]
    for project in all_projects:
        if project not in results['samples_per_project']:
            results['samples_per_project'][project] = 0
    
    # Responders vs non-responders
    cursor.execute(f"""
        SELECT 
            SUM(CASE WHEN response = 'yes' THEN 1 ELSE 0 END) as responders,
            SUM(CASE WHEN response = 'no' THEN 1 ELSE 0 END) as non_responders
        FROM samples 
        WHERE {base_filter}
    """)
    row = cursor.fetchone()
    results['responders'] = row[0]
    results['non_responders'] = row[1]
    
    # Males vs females
    cursor.execute(f"""
        SELECT 
            SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) as males,
            SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) as females
        FROM samples 
        WHERE {base_filter}
    """)
    row = cursor.fetchone()
    results['males'] = row[0]
    results['females'] = row[1]
    
    if close_conn:
        conn.close()
    
    return results


def format_baseline_results(results: Dict[str, Any]) -> str:
    """
    Format baseline analysis results as a readable text report.
    
    Args:
        results: Dictionary from analyze_baseline_subset()
        
    Returns:
        Formatted string report.
    """
    total = results['total_samples']
    
    lines = [
        "Melanoma PBMC Baseline Analysis (Miraclib Treatment)",
        "=" * 55,
        "",
        f"Total samples matching criteria: {results['total_samples']}",
        f"Total unique subjects: {results['unique_subjects']}",
        "",
        "4a. Samples per project:",
    ]
    
    for project in sorted(results['samples_per_project'].keys()):
        count = results['samples_per_project'][project]
        if count > 0:
            lines.append(f"  - {project}: {count} samples")
        else:
            lines.append(f"  - {project}: 0 samples (no data matching criteria)")
    
    lines.extend([
        "",
        "4b. Subject response distribution:",
        f"  - Responders: {results['responders']} subjects ({results['responders']/total*100:.1f}%)",
        f"  - Non-responders: {results['non_responders']} subjects ({results['non_responders']/total*100:.1f}%)",
        "",
        "4c. Subject sex distribution:",
        f"  - Males: {results['males']} subjects ({results['males']/total*100:.1f}%)",
        f"  - Females: {results['females']} subjects ({results['females']/total*100:.1f}%)",
    ])
    
    return "\n".join(lines)


def save_baseline_results(results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """
    Save baseline analysis results to text file.
    
    Args:
        results: Dictionary from analyze_baseline_subset()
        output_path: Path for output file. Uses default if not provided.
        
    Returns:
        Path to saved file.
    """
    if output_path is None:
        output_path = get_project_root() / 'outputs' / 'part4_results.txt'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    formatted = format_baseline_results(results)
    
    with open(output_path, 'w') as f:
        f.write(formatted)
    
    print(f"Baseline analysis results saved to {output_path}")
    
    return output_path


def run_part2(conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """
    Run Part 2 analysis: Calculate and save cell frequencies.
    
    Args:
        conn: SQLite connection. Creates new connection if not provided.
        
    Returns:
        DataFrame with cell frequencies.
    """
    print("\n" + "=" * 60)
    print("PART 2: Cell Frequency Summary")
    print("=" * 60)
    
    df = calculate_cell_frequencies(conn)
    
    # Validate
    print("\nValidating percentages sum to 100%...")
    if validate_frequencies(df):
        print("✓ Validation passed")
    
    # Save
    save_summary_table(df)
    
    # Print sample output
    print("\nSample output (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    return df


def run_part4(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """
    Run Part 4 analysis: Baseline subset analysis.
    
    Args:
        conn: SQLite connection. Creates new connection if not provided.
        
    Returns:
        Dictionary with analysis results.
    """
    print("\n" + "=" * 60)
    print("PART 4: Baseline Subset Analysis")
    print("=" * 60)
    
    results = analyze_baseline_subset(conn)
    save_baseline_results(results)
    
    # Print results
    print("\n" + format_baseline_results(results))
    
    return results


if __name__ == "__main__":
    # Run analyses when executed directly
    from .database import initialize_database
    
    conn = initialize_database()
    
    run_part2(conn)
    run_part4(conn)
    
    conn.close()
    print("\nAnalysis complete.")

