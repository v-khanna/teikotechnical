#!/usr/bin/env python3
"""
Clinical Trial Data Analysis - Main Entry Point

This script runs all four parts of the analysis:
- Part 1: Database initialization
- Part 2: Cell frequency summary
- Part 3: Statistical analysis (responders vs non-responders)
- Part 4: Baseline subset analysis

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.database import initialize_database, get_db_path
from src.analysis import run_part2, run_part4
from src.statistics import run_part3


def main():
    """Run the complete clinical trial analysis pipeline."""
    print("=" * 70)
    print("CLINICAL TRIAL DATA ANALYSIS")
    print("=" * 70)
    
    # Part 1: Initialize database
    print("\n" + "=" * 70)
    print("PART 1: Database Initialization")
    print("=" * 70)
    
    conn = initialize_database()
    
    # Part 2: Cell frequency summary
    run_part2(conn)
    
    # Part 3: Statistical analysis
    run_part3(conn)
    
    # Part 4: Baseline subset analysis
    run_part4(conn)
    
    # Cleanup
    conn.close()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files generated:")
    print(f"  - outputs/part2_summary_table.csv (52,500 rows)")
    print(f"  - outputs/part3_boxplots.png")
    print(f"  - outputs/part3_statistics.csv")
    print(f"  - outputs/part4_results.txt")
    print(f"\nDatabase: data/clinical_trial.db")
    print("\nTo launch the dashboard:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

