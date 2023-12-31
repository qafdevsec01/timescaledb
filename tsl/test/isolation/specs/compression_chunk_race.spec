# This file and its contents are licensed under the Timescale License.
# Please see the included NOTICE for copyright information and
# LICENSE-TIMESCALE for a copy of the license.

###
# Test the execution of two compression jobs in parallel
###

setup {
   CREATE TABLE sensor_data (
   time timestamptz not null,
   sensor_id integer not null,
   cpu double precision null,
   temperature double precision null);

   -- Create large chunks that take a long time to compress
   SELECT FROM create_hypertable('sensor_data','time', chunk_time_interval => INTERVAL '14 days');

   INSERT INTO sensor_data
   SELECT
   time + (INTERVAL '1 minute' * random()) AS time,
   sensor_id,
   random() AS cpu,
   random()* 100 AS temperature
   FROM
   generate_series('2022-01-01', '2022-01-15', INTERVAL '1 minute') AS g1(time),
   generate_series(1, 50, 1) AS g2(sensor_id)
   ORDER BY time;

   SELECT count(*) FROM sensor_data;
   
   ALTER TABLE sensor_data SET (timescaledb.compress, timescaledb.compress_segmentby = 'sensor_id');
}

teardown {
   DROP TABLE sensor_data;
}


session "s1"
setup {
   SET client_min_messages=ERROR; -- Suppress chunk "_hyper_XXX_chunk" is not compressed messages 
}

step "s1_compress" {
   SELECT compression_status FROM chunk_compression_stats('sensor_data');
   SELECT count(*) FROM (SELECT compress_chunk(i, if_not_compressed => true) FROM show_chunks('sensor_data') i) i;
   SELECT compression_status FROM chunk_compression_stats('sensor_data');
   SELECT count(*) FROM sensor_data;
}

step "s1_decompress" {
   SELECT count(*) FROM (SELECT decompress_chunk(i, if_compressed => true) FROM show_chunks('sensor_data') i) i;
   SELECT compression_status FROM chunk_compression_stats('sensor_data');
   SELECT count(*) FROM sensor_data;
}

session "s2"
setup {
   SET client_min_messages=ERROR; -- Suppress chunk "_hyper_XXX_chunk" is not compressed messages 
}

step "s2_compress" {
   SELECT compression_status FROM chunk_compression_stats('sensor_data');
   SELECT count(*) FROM (SELECT compress_chunk(i, if_not_compressed => true) FROM show_chunks('sensor_data') i) i;
   SELECT compression_status FROM chunk_compression_stats('sensor_data');
   SELECT count(*) FROM sensor_data;
   RESET client_min_messages;
}

step "s2_decompress" {
   SELECT count(*) FROM (SELECT decompress_chunk(i, if_compressed => true) FROM show_chunks('sensor_data') i) i;
   SELECT compression_status FROM chunk_compression_stats('sensor_data');
   SELECT count(*) FROM sensor_data;
   RESET client_min_messages;
}

session "s3"
step "s3_lock_compression" {
    SELECT debug_waitpoint_enable('compress_chunk_impl_start');
}

step "s3_lock_decompression" {
    SELECT debug_waitpoint_enable('decompress_chunk_impl_start');
}

step "s3_unlock_compression" {
    -- Ensure that we are waiting on our debug waitpoint and one chunk
    -- Note: The OIDs of the advisory locks are based on the hash value of the lock name (see debug_point_init())
    --       compress_chunk_impl_start = 3379597659.
    -- 'SELECT relation::regclass, ....' can not be used, because it returns a field with a variable length
    SELECT locktype, mode, granted, objid FROM pg_locks WHERE not granted AND (locktype = 'advisory' or relation::regclass::text LIKE '%chunk') ORDER BY relation, locktype, mode, granted;
    SELECT debug_waitpoint_release('compress_chunk_impl_start');
}

step "s3_unlock_decompression" {
    -- Ensure that we are waiting on our debug waitpoint and one chunk
    -- Note: The OIDs of the advisory locks are based on the hash value of the lock name (see debug_point_init())
    --       decompress_chunk_impl_start = 2415054640.
    -- 'SELECT relation::regclass, ....' can not be used, because it returns a field with a variable length
    SELECT locktype, mode, granted, objid FROM pg_locks WHERE not granted AND (locktype = 'advisory' or relation::regclass::text LIKE '%chunk') ORDER BY relation, locktype, mode, granted;
    SELECT debug_waitpoint_release('decompress_chunk_impl_start');
}

permutation "s3_lock_compression" "s3_lock_decompression" "s1_compress" "s2_compress" (s1_compress) "s3_unlock_compression" "s1_decompress" "s2_decompress" (s1_decompress) "s3_unlock_decompression" 

