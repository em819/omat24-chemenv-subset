#!/usr/bin/env python3


import duckdb
import time
import os
import pandas as pd

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK (or set the environment variables) <<<
#
#   CHEMENV_FREQ_DB   INPUT. The freq_chemenv.duckdb file written by stage 02.
#                     The default assumes this script runs in the SAME directory
#                     stage 02 ran in; otherwise give an absolute path:
#                       export CHEMENV_FREQ_DB=/data/work/selection/freq_chemenv.duckdb
#
#   TOKEN_VEC_DB      OUTPUT. The material-vector database. Stage 04 reads it as
#                     $V10_TOKEN_VEC_DB. The 1000 intermediate
#                     material_vectors_batch*.parquet shards are written to the
#                     CURRENT WORKING DIRECTORY, so run this where there is room.
#
#   DUCK_THREADS / DUCK_MEMORY   DuckDB resources. DUCK_MEMORY is the one to
#                     lower on a smaller machine.
#   DUCK_TEMP_DIR     Spill directory. DuckDB spills here rather than failing
#                     when DUCK_MEMORY is tight; the spill can reach several
#                     hundred GB on the full corpus.
# ============================================================================
chemenv_db   = os.environ.get('CHEMENV_FREQ_DB', 'freq_chemenv.duckdb')
TOKEN_VEC_DB = os.environ.get('TOKEN_VEC_DB', 'material_vectors.duckdb')
DUCK_THREADS = int(os.environ.get('DUCK_THREADS', '128'))
DUCK_MEMORY  = os.environ.get('DUCK_MEMORY', '400GB')
TEMP_DIR     = os.environ.get('DUCK_TEMP_DIR', './duckdb_temp')
# ============================================================================

if not os.path.isfile(chemenv_db):
    raise FileNotFoundError(
        f"{chemenv_db} not found. Set CHEMENV_FREQ_DB to the freq_chemenv.duckdb "
        f"written by stage 02.")
os.makedirs(TEMP_DIR, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Connect to new database for vectors
time_connect_start = time.time()
conn_vectors = duckdb.connect(TOKEN_VEC_DB,
        config={'threads' : DUCK_THREADS,
                'memory_limit' : DUCK_MEMORY,
                'preserve_insertion_order' : False})
conn_vectors.execute(f"ATTACH '{chemenv_db}' AS original_db (READ_ONLY)")

conn_vectors.execute(f"SET temp_directory='{TEMP_DIR}'")

time_connect_end = time.time()

print(f"Time for connecting to new db : {time_connect_end - time_connect_start} s", flush=True)

# Check source data dimensions
print("Analyzing source data...")
total_rows = conn_vectors.execute("SELECT COUNT(*) FROM original_db.data").fetchone()[0]
unique_mat_ids = conn_vectors.execute("SELECT COUNT(DISTINCT mat_id) FROM original_db.data").fetchone()[0]
unique_env_ids = conn_vectors.execute("SELECT COUNT(DISTINCT env_id) FROM original_db.data").fetchone()[0]

print(f"Total rows (env_ids): {total_rows:,}")
print(f"Unique mat_ids: {unique_mat_ids:,}")
print(f"Unique env_ids: {unique_env_ids:,}")

#Save mapping of env_id to integer index
time_map_start = time.time()
conn_vectors.execute("""
  CREATE TABLE environment_mapping AS
  SELECT
    env_id,
    (ROW_NUMBER() OVER (ORDER BY env_id) - 1)::INTEGER as env_index
FROM (SELECT DISTINCT env_id FROM original_db.data)
ORDER BY env_index
""")
mapping_rows = conn_vectors.execute("SELECT COUNT(*) FROM environment_mapping").fetchone()[0]
print(f"Rows in environment_mapping: {mapping_rows}", flush=True)

time_map_end = time.time()
print(f"Time for saving mapping of env_id -> index: {time_map_end - time_map_start} s", flush=True)

#a third column freq (number of env_ids in mat_id)
time_count_envs_start = time.time()
conn_vectors.execute("""
    CREATE TABLE env_counts AS
    SELECT
        mat_id,
        env_id,
        COUNT(*) as count
    FROM original_db.data
    GROUP BY mat_id, env_id
    ORDER BY mat_id, env_id
""")

env_counts_rows = conn_vectors.execute("SELECT COUNT(*) FROM env_counts").fetchone()[0]
time_count_envs_end = time.time()
print(f"Time for counting env_ids per mat_id: {time_count_envs_end - time_count_envs_start} s", flush=True)
print(f"Rows in env_counts: {env_counts_rows:,}", flush=True)

#conn_vectors.execute("""
#        mat_id,
#        NTILE(10000) OVER (ORDER BY mat_id) as batch_num
#""")
# Get batch count
## Also useful: index on mat_id for the batching


num_batches = 1000

time_vec_start = time.time()
for batch in range(num_batches):
    print(f"Processing batch {batch}/{num_batches}", flush=True)

    time_create_batch_start = time.time()
    conn_vectors.execute(f"""
        CREATE OR REPLACE TEMP TABLE batch_data AS
        SELECT
            ec.mat_id,
            em.env_index,
            ec.count
        FROM env_counts ec
        JOIN environment_mapping em ON ec.env_id = em.env_id
        WHERE hash(ec.mat_id) % {num_batches} = {batch}
    """)
    batch_rows = conn_vectors.execute("SELECT COUNT(*) FROM batch_data").fetchone()[0]
    time_create_batch_end = time.time()
    print(f"Time for creating batch table ({batch_rows} elements): {time_create_batch_end - time_create_batch_start} s", flush=True)

    time_process_batch_start = time.time()
    conn_vectors.execute(f"""
        COPY (
            SELECT
                mat_id,
                LIST(env_index ORDER BY env_index) as indices,
                LIST(count ORDER BY env_index) as counts
            FROM batch_data
            GROUP BY mat_id
        ) TO 'material_vectors_batch_{batch:04d}.parquet'
        (FORMAT PARQUET, COMPRESSION 'zstd')
    """)
    time_process_batch_end = time.time()
    print(f"Time to process batch: {time_process_batch_end - time_process_batch_start} s", flush=True)

#    conn_vectors.execute(f"""
#    """)

#conn_vectors.execute("""
#""")
time_vec_end = time.time()
print(f"Time for creating the token vectors: {time_vec_end - time_vec_start} s", flush=True)

# Detach source database
conn_vectors.execute("DETACH original_db")

conn_vectors.execute("CREATE TABLE material_vectors AS SELECT * FROM read_parquet('material_vectors_batch*.parquet')")
#Normalize the sparse token vectors to 1
time_norm_start = time.time()
#########################################
####Normalized token vectors#############
#########################################
#conn_vectors.execute("""
#        mat_id,
#        ROW_NUMBER() OVER (ORDER BY mat_id) - 1 AS mat_index,
#        indices,
#            counts,
#            x -> x::DOUBLE / SQRT(LIST_SUM(LIST_TRANSFORM(counts, y -> y::DOUBLE * y::DOUBLE)))
#""")

#########################################
####Non-normalized token vectors#########
#########################################
conn_vectors.execute("""
    CREATE OR REPLACE TABLE material_vectors_temp AS
    SELECT
        mat_id,
        ROW_NUMBER() OVER (ORDER BY mat_id) - 1 AS mat_index,
        indices,
        counts
    FROM material_vectors
""")

conn_vectors.execute("DROP TABLE material_vectors")
conn_vectors.execute("ALTER TABLE material_vectors_temp RENAME TO material_vectors")

time_norm_end = time.time()
print(f"Time to normalize token vectors: {time_norm_end - time_norm_start} s", flush=True)

conn_vectors.execute("CREATE INDEX IF NOT EXISTS idx_mat_vecs_mat_id ON material_vectors(mat_id)")

# Check for duplicates
result = conn_vectors.execute("""
    SELECT mat_id, COUNT(*) as cnt 
    FROM material_vectors 
    GROUP BY mat_id 
    HAVING COUNT(*) > 1
""").fetchall()

if result:
    print(f"Warning: Found duplicate mat_ids: {result}", flush=True)

# Verify results
print("Sample of the new table:", flush=True)
print(conn_vectors.execute("SELECT * FROM material_vectors LIMIT 10").fetchdf(), flush=True)

conn_vectors.close()
