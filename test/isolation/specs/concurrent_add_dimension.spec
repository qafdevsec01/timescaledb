# This file and its contents are licensed under the Apache License 2.0.
# Please see the included NOTICE for copyright information and
# LICENSE-APACHE for a copy of the license.

setup {
  DROP TABLE IF EXISTS dim_test;
  CREATE TABLE dim_test(time TIMESTAMPTZ, device int, device2 int);
  SELECT table_name FROM create_hypertable('dim_test', 'time', chunk_time_interval => INTERVAL '1 day');
  INSERT INTO dim_test VALUES ('2004-10-10 00:00:00+00', 1, 1);
}

teardown {
  DROP TABLE dim_test;
}

session "s1"
step "s1_wp_enable"        { SELECT debug_waitpoint_enable('add_dimension_ht_lock'); }
step "s1_wp_release"       { SELECT debug_waitpoint_release('add_dimension_ht_lock'); }
step "s1_add_dimension"	   { SELECT column_name FROM add_dimension('dim_test', 'device', 2); }
step "s1_create_chunk"     { INSERT INTO dim_test VALUES ('2004-10-20 00:00:00+00', 1, 2); }

session "s2"
step "s2_add_dimension"	   { SELECT column_name FROM add_dimension('dim_test', 'device', 1); }
step "s2_add_dimension2"   { SELECT column_name FROM add_dimension('dim_test', 'device2', 1); }

session "s3"
step "s3_wp_enable"        { SELECT debug_waitpoint_enable('add_dimension_ht_lock'); }
step "s3_wp_release"       { SELECT debug_waitpoint_release('add_dimension_ht_lock'); }
step "s3_chunk_wp_enable"  { SELECT debug_waitpoint_enable('chunk_create_for_point'); }
step "s3_chunk_wp_release" { SELECT debug_waitpoint_release('chunk_create_for_point'); }

step "s3_query"            {
	SELECT count(*)
	FROM _timescaledb_catalog.chunk c
	INNER JOIN _timescaledb_catalog.hypertable h ON (c.hypertable_id = h.id)
	INNER JOIN _timescaledb_catalog.dimension td ON (h.id = td.hypertable_id)
	INNER JOIN _timescaledb_catalog.dimension_slice ds ON (ds.dimension_id = td.id)
	INNER JOIN _timescaledb_catalog.chunk_constraint cc ON (cc.dimension_slice_id = ds.id AND cc.chunk_id = c.id)
	WHERE h.table_name = 'dim_test';
}

# Test concurrent add_dimension() call with existing data
#
permutation "s3_wp_enable" "s1_add_dimension" "s2_add_dimension" "s3_wp_release" "s3_query"

# Test concurrent chunk creation during add_dimension() call
#
permutation "s3_chunk_wp_enable" "s1_create_chunk" "s2_add_dimension2" "s3_chunk_wp_release" "s3_query"
