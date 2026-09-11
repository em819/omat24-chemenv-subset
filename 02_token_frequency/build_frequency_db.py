"""Load the per-site token parquet files into DuckDB and build three tables.

    data                     every site, one row each
    chemenv_frequency_table  token -> how often it occurs
    material_elements        material -> list of its elements

Stages 03 and 04 both read the resulting database, so keep it.

Paths and resource limits come from the environment:
    CHEMENV_PARQUET_GLOB   where stage 01b wrote its parquet files
    CHEMENV_FREQ_DB        output database
    DUCK_THREADS, DUCK_MEMORY, DUCK_TEMP_DIR
Lower the memory limit on a smaller machine; DuckDB spills to the temp
directory instead of failing.
"""

import duckdb
import os
import time
import pandas as pd

pd.set_option('display.max_rows', None)

PARQUET_GLOB = os.environ.get('CHEMENV_PARQUET_GLOB',
                              '../parquet_file_collection/total_chemenv*.parquet')
FREQ_DB      = os.environ.get('CHEMENV_FREQ_DB', 'freq_chemenv.duckdb')
DUCK_THREADS = int(os.environ.get('DUCK_THREADS', '128'))
DUCK_MEMORY  = os.environ.get('DUCK_MEMORY', '400GB')
TEMP_DIR     = os.environ.get('DUCK_TEMP_DIR', './duckdb_temp')

os.makedirs(TEMP_DIR, exist_ok=True)
conn = duckdb.connect(FREQ_DB,
        config={'threads': DUCK_THREADS,
                'memory_limit': DUCK_MEMORY,
                'preserve_insertion_order': False})
conn.execute(f"SET temp_directory='{TEMP_DIR}'")
print(f'Reading parquet from {PARQUET_GLOB}', flush=True)
print(f'Writing {FREQ_DB} (threads={DUCK_THREADS}, memory={DUCK_MEMORY})', flush=True)

# Read every parquet file into one table.
t_create_data_start = time.time()
conn.execute(f"CREATE TABLE data AS SELECT * FROM read_parquet('{PARQUET_GLOB}')")
t_create_data_end = time.time()
print(f'Time to unite all chemenv data : {t_create_data_end - t_create_data_start} s', flush=True)

# Count how often each token occurs.
t_create_freq_start = time.time()
conn.execute("""
    CREATE TABLE chemenv_frequency_table AS
    SELECT 
        env_id,
        COUNT(*) as frequency
    FROM data
    GROUP BY env_id
    ORDER BY frequency DESC
""")
t_create_freq_end = time.time()
print(f'Time to create freq table: {t_create_freq_end - t_create_freq_start} s', flush=True)

t_create_freq_idx_start = time.time()
conn.execute("CREATE INDEX idx_freq_env_id ON chemenv_frequency_table(env_id)")
t_create_freq_idx_end = time.time()
print(f'Time to create indices for freq table : {t_create_freq_idx_end - t_create_freq_idx_start} s')

# Which elements each material contains.
t_create_elems_start = time.time()
conn.execute("""
        CREATE TABLE material_elements 
        AS
        SELECT mat_id, LIST(DISTINCT central_element ORDER BY central_element) AS elements
        FROM data
        GROUP BY mat_id
        """)
t_create_elems_end = time.time()
print(f'Time to create elements table: {t_create_elems_end - t_create_elems_start} s', flush=True)
print(conn.execute("SELECT * FROM chemenv_frequency_table LIMIT 100000").df())

conn.close()
