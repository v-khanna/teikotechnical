"""
Database module for clinical trial data management.

Handles SQLite database creation, schema definition, and data loading.
Uses a simplified 2-table design optimized for the analysis requirements.
"""

import sqlite3
import csv
from pathlib import Path
from typing import Optional


# Cell population names
CELL_POPULATIONS = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_db_path() -> Path:
    """Get the default database path."""
    return get_project_root() / 'data' / 'clinical_trial.db'


def get_csv_path() -> Path:
    """Get the default CSV data path."""
    return get_project_root() / 'data' / 'cell-count.csv'


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection.
    
    Args:
        db_path: Path to database file. Uses default if not provided.
        
    Returns:
        SQLite connection object.
    """
    if db_path is None:
        db_path = get_db_path()
    return sqlite3.connect(db_path)


def create_schema(conn: sqlite3.Connection) -> None:
    """
    Create the database schema with two tables.
    
    Table 1: samples - Contains all sample metadata (10,500 rows)
    Table 2: cell_counts - Contains cell population counts in long format (52,500 rows)
    
    Args:
        conn: SQLite connection object.
    """
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS cell_counts")
    cursor.execute("DROP TABLE IF EXISTS samples")
    
    # Create samples table with all metadata
    cursor.execute("""
        CREATE TABLE samples (
            sample_id VARCHAR(20) PRIMARY KEY,
            project VARCHAR(10) NOT NULL,
            subject VARCHAR(20) NOT NULL,
            condition VARCHAR(20) NOT NULL,
            age INTEGER NOT NULL,
            sex CHAR(1) NOT NULL,
            treatment VARCHAR(20) NOT NULL,
            response VARCHAR(3),
            sample_type VARCHAR(10) NOT NULL,
            time_from_treatment_start INTEGER NOT NULL
        )
    """)
    
    # Create cell_counts table in long format
    cursor.execute("""
        CREATE TABLE cell_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id VARCHAR(20) NOT NULL,
            population VARCHAR(20) NOT NULL,
            count INTEGER NOT NULL,
            FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
            UNIQUE(sample_id, population)
        )
    """)
    
    # Create indexes for frequently queried columns
    cursor.execute("CREATE INDEX idx_samples_project ON samples(project)")
    cursor.execute("CREATE INDEX idx_samples_subject ON samples(subject)")
    cursor.execute("CREATE INDEX idx_samples_condition ON samples(condition)")
    cursor.execute("CREATE INDEX idx_samples_treatment ON samples(treatment)")
    cursor.execute("CREATE INDEX idx_samples_sample_type ON samples(sample_type)")
    cursor.execute("CREATE INDEX idx_samples_time ON samples(time_from_treatment_start)")
    cursor.execute("CREATE INDEX idx_samples_response ON samples(response)")
    cursor.execute("CREATE INDEX idx_cell_counts_sample ON cell_counts(sample_id)")
    cursor.execute("CREATE INDEX idx_cell_counts_population ON cell_counts(population)")
    
    conn.commit()
    print("Database schema created successfully.")


def load_data(conn: sqlite3.Connection, csv_path: Optional[Path] = None) -> None:
    """
    Load data from CSV file into the database.
    
    Transforms the wide-format CSV into the normalized database schema.
    
    Args:
        conn: SQLite connection object.
        csv_path: Path to CSV file. Uses default if not provided.
    """
    if csv_path is None:
        csv_path = get_csv_path()
    
    cursor = conn.cursor()
    
    samples_inserted = 0
    cell_counts_inserted = 0
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Insert into samples table
            # Handle empty response values (healthy patients have no response)
            response = row['response'] if row['response'] else None
            
            cursor.execute("""
                INSERT INTO samples (
                    sample_id, project, subject, condition, age, sex, 
                    treatment, response, sample_type, time_from_treatment_start
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['sample'],
                row['project'],
                row['subject'],
                row['condition'],
                int(row['age']),
                row['sex'],
                row['treatment'],
                response,
                row['sample_type'],
                int(row['time_from_treatment_start'])
            ))
            samples_inserted += 1
            
            # Insert cell counts into cell_counts table (long format)
            for population in CELL_POPULATIONS:
                cursor.execute("""
                    INSERT INTO cell_counts (sample_id, population, count)
                    VALUES (?, ?, ?)
                """, (
                    row['sample'],
                    population,
                    int(row[population])
                ))
                cell_counts_inserted += 1
    
    conn.commit()
    print(f"Data loaded successfully:")
    print(f"  - Samples inserted: {samples_inserted}")
    print(f"  - Cell count records inserted: {cell_counts_inserted}")


def validate_data(conn: sqlite3.Connection) -> bool:
    """
    Validate that data was loaded correctly.
    
    Args:
        conn: SQLite connection object.
        
    Returns:
        True if validation passes, False otherwise.
    """
    cursor = conn.cursor()
    
    # Check sample count
    cursor.execute("SELECT COUNT(*) FROM samples")
    sample_count = cursor.fetchone()[0]
    
    # Check cell counts
    cursor.execute("SELECT COUNT(*) FROM cell_counts")
    cell_count = cursor.fetchone()[0]
    
    # Check unique populations
    cursor.execute("SELECT COUNT(DISTINCT population) FROM cell_counts")
    population_count = cursor.fetchone()[0]
    
    # Validate expected counts
    expected_samples = 10500
    expected_cell_counts = 52500  # 10500 * 5 populations
    expected_populations = 5
    
    valid = True
    
    if sample_count != expected_samples:
        print(f"WARNING: Expected {expected_samples} samples, found {sample_count}")
        valid = False
    else:
        print(f"✓ Sample count verified: {sample_count}")
    
    if cell_count != expected_cell_counts:
        print(f"WARNING: Expected {expected_cell_counts} cell counts, found {cell_count}")
        valid = False
    else:
        print(f"✓ Cell count records verified: {cell_count}")
    
    if population_count != expected_populations:
        print(f"WARNING: Expected {expected_populations} populations, found {population_count}")
        valid = False
    else:
        print(f"✓ Population count verified: {population_count}")
    
    return valid


def initialize_database(db_path: Optional[Path] = None, csv_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Initialize the database: create schema and load data.
    
    This is the main entry point for database setup.
    
    Args:
        db_path: Path to database file. Uses default if not provided.
        csv_path: Path to CSV file. Uses default if not provided.
        
    Returns:
        SQLite connection object.
    """
    if db_path is None:
        db_path = get_db_path()
    
    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database if it exists
    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing database at {db_path}")
    
    conn = get_connection(db_path)
    
    print(f"\nInitializing database at {db_path}")
    print("-" * 50)
    
    create_schema(conn)
    load_data(conn, csv_path)
    
    print("\nValidating data...")
    print("-" * 50)
    validate_data(conn)
    
    return conn


if __name__ == "__main__":
    # Run database initialization when executed directly
    conn = initialize_database()
    conn.close()
    print("\nDatabase initialization complete.")

