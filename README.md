# Clinical Trial Data Analysis

Analysis pipeline for clinical trial immune cell population data, examining how drug treatments affect immune cell distributions in cancer patients.

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/[username]/teikotechnical.git
cd teikotechnical

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python main.py

# Launch the dashboard
streamlit run dashboard/app.py
```

### Running in GitHub Codespaces

1. Open the repository in Codespaces
2. Wait for the environment to initialize
3. Run in terminal:
   ```bash
   pip install -r requirements.txt
   python main.py
   streamlit run dashboard/app.py
   ```
4. Click the "Open in Browser" link when Streamlit starts

## Dashboard

**Live Dashboard:** [https://teikotechnical.streamlit.app](https://teikotechnical.streamlit.app)

*Note: Replace with actual deployed URL*

## Project Structure

```
teikotechnical/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Main entry point - runs all analyses
├── data/
│   └── cell-count.csv        # Input data (10,500 samples)
├── src/
│   ├── database.py           # Part 1: Database schema and loading
│   ├── analysis.py           # Parts 2 & 4: Frequency and subset analysis
│   └── statistics.py         # Part 3: Statistical tests and visualization
├── outputs/
│   ├── part2_summary_table.csv   # Cell frequencies (52,500 rows)
│   ├── part3_boxplots.png        # Comparison visualizations
│   ├── part3_statistics.csv      # Statistical test results
│   └── part4_results.txt         # Baseline characteristics
└── dashboard/
    └── app.py                # Streamlit dashboard
```

## Database Schema

### Design Choice: Simplified 2-Table Schema

I chose a simplified 2-table design optimized for the specific analysis requirements:

```
┌─────────────────────────────────────────────────────────────┐
│                         samples                              │
├─────────────────────────────────────────────────────────────┤
│ sample_id (PK)       VARCHAR(20)  - Unique sample identifier│
│ project              VARCHAR(10)  - Project ID              │
│ subject              VARCHAR(20)  - Subject ID              │
│ condition            VARCHAR(20)  - Disease condition       │
│ age                  INTEGER      - Patient age             │
│ sex                  CHAR(1)      - M or F                  │
│ treatment            VARCHAR(20)  - Treatment name          │
│ response             VARCHAR(3)   - yes/no/NULL             │
│ sample_type          VARCHAR(10)  - PBMC or WB              │
│ time_from_treatment_start INTEGER - Days from start         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:5
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       cell_counts                            │
├─────────────────────────────────────────────────────────────┤
│ id (PK)              INTEGER      - Auto-increment          │
│ sample_id (FK)       VARCHAR(20)  - Links to samples        │
│ population           VARCHAR(20)  - Cell type name          │
│ count                INTEGER      - Cell count              │
│ UNIQUE(sample_id, population)                               │
└─────────────────────────────────────────────────────────────┘
```

### Rationale

**Why 2 tables instead of 4 (normalized)?**

1. **Query Efficiency**: The cell_counts table in long format directly supports Part 2's frequency calculations and Part 3's statistical analysis without complex joins.

2. **Simplicity**: Fewer tables mean simpler queries, less maintenance, and faster development for the analysis requirements.

3. **Sufficient Normalization**: Subject and project data are still separated from cell counts, preventing the worst redundancy issues.

### Indexes

```sql
-- Sample filtering (used in Parts 3 & 4)
CREATE INDEX idx_samples_condition ON samples(condition);
CREATE INDEX idx_samples_treatment ON samples(treatment);
CREATE INDEX idx_samples_sample_type ON samples(sample_type);
CREATE INDEX idx_samples_time ON samples(time_from_treatment_start);
CREATE INDEX idx_samples_response ON samples(response);

-- Cell count lookups
CREATE INDEX idx_cell_counts_sample ON cell_counts(sample_id);
CREATE INDEX idx_cell_counts_population ON cell_counts(population);
```

### Scalability Considerations

**Scaling to 100+ projects:**
- Project filtering uses indexed `project` column
- No schema changes needed - just add rows
- Could add separate `projects` table for project metadata (PI, dates, etc.)

**Scaling to 1000+ samples:**
- Indexes on frequently filtered columns ensure O(log n) lookups
- Long format in cell_counts scales linearly (5 rows per sample)
- Could partition cell_counts by sample_id range for very large datasets

**Supporting various analytics:**
- Long format enables easy aggregations by population
- Adding new cell populations requires no schema changes
- Can add computed columns or materialized views for common queries
- Results tables could cache expensive computations

**Future extensions:**
- Add `analysis_runs` table to track analysis versions
- Add `treatments` metadata table for treatment details
- Add `conditions` metadata table for condition hierarchies
- Implement time-series analysis with proper temporal indexing

## Analysis Components

### Part 1: Database Management
- Creates SQLite database with optimized schema
- Loads 10,500 samples from CSV
- Transforms cell counts to long format (52,500 records)

### Part 2: Cell Frequency Summary
- Calculates total cell count per sample
- Computes relative frequency (%) for each population
- Validates percentages sum to 100%
- **Output:** `outputs/part2_summary_table.csv`

### Part 3: Statistical Analysis
- **Filter:** Melanoma patients + Miraclib treatment + PBMC samples
- **Samples:** 1,968 (993 responders, 975 non-responders)
- **Test:** Mann-Whitney U (non-parametric)
- **Correction:** Bonferroni (α = 0.01 per test)
- **Output:** `outputs/part3_boxplots.png`, `outputs/part3_statistics.csv`

### Part 4: Baseline Subset Analysis
- **Filter:** Melanoma + PBMC + Baseline (Day 0) + Miraclib
- **Samples:** 656
- Reports project distribution, response rates, sex distribution
- **Output:** `outputs/part4_results.txt`

## Key Findings

### Part 3 Results
No statistically significant differences were found between responders and non-responders after Bonferroni correction for any of the 5 cell populations.

| Population | Resp. Mean | Non-Resp. Mean | Adj. P-value | Significant |
|------------|------------|----------------|--------------|-------------|
| b_cell     | 9.80%      | 10.00%         | 0.279        | No          |
| cd8_t_cell | 24.88%     | 24.94%         | 1.000        | No          |
| cd4_t_cell | 30.54%     | 29.90%         | 0.067        | No          |
| nk_cell    | 14.84%     | 15.07%         | 0.605        | No          |
| monocyte   | 19.94%     | 20.08%         | 0.816        | No          |

### Part 4 Results
- Total baseline samples: 656
- Project distribution: prj1 (384), prj3 (272), prj2 (0)
- Response: 50.5% responders, 49.5% non-responders
- Sex: 52.4% male, 47.6% female

## Technical Notes

### Column Name Discrepancy
The task description uses different column names than the actual CSV:

| Task Says | CSV Has | Used in Code |
|-----------|---------|--------------|
| sample_id | sample  | sample       |
| indication| condition | condition  |
| gender    | sex     | sex          |

### Dependencies
- pandas 2.1.4
- numpy 1.26.2
- scipy 1.11.4
- matplotlib 3.8.2
- seaborn 0.13.0
- streamlit 1.29.0

## Code Design

### Separation of Concerns
- **database.py**: Data access layer - schema, loading, connections
- **analysis.py**: Business logic for Parts 2 & 4
- **statistics.py**: Statistical analysis for Part 3
- **app.py**: Presentation layer (dashboard)

### Design Principles
1. **Single Responsibility**: Each module handles one aspect
2. **Dependency Injection**: Functions accept optional connections for testing
3. **Fail-Safe Defaults**: Functions use sensible defaults via Path utilities
4. **Validation**: Data integrity checks at load time and analysis time

## License

MIT License

