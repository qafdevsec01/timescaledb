/*
 * This file and its contents are licensed under the Apache License 2.0.
 * Please see the included NOTICE for copyright information and
 * LICENSE-APACHE for a copy of the license.
 */
#include <postgres.h>
#include <catalog/pg_type.h>
#include <nodes/nodeFuncs.h>
#include <nodes/nodes.h>
#include <nodes/pathnodes.h>
#include <nodes/pg_list.h>
#include <optimizer/appendinfo.h>
#include <optimizer/cost.h>
#include <optimizer/optimizer.h>
#include <optimizer/pathnode.h>
#include <optimizer/paths.h>
#include <optimizer/planner.h>
#include <optimizer/prep.h>
#include <optimizer/tlist.h>
#include <parser/parse_func.h>
#include <utils/lsyscache.h>

#include "debug_assert.h"
#include "partialize.h"
#include "planner.h"
#include "nodes/print.h"
#include "extension_constants.h"
#include "utils.h"
#include "estimate.h"
#include "nodes/chunk_append/chunk_append.h"
#include "import/planner.h"

#define TS_PARTIALFN "partialize_agg"

typedef struct PartializeWalkerState
{
	bool found_partialize;
	bool found_non_partial_agg;
	bool looking_for_agg;
	Oid fnoid;
	PartializeAggFixAggref fix_aggref;
} PartializeWalkerState;

/*
 * Look for the partialize function in a target list and mark the wrapped
 * aggregate as a partial aggregate.
 *
 * The partialize function is an expression of the form:
 *
 * _timescaledb_internal.partialize_agg(avg(temp))
 *
 * where avg(temp) can be replaced by any aggregate that can be partialized.
 *
 * When such an expression is found, this function will mark the Aggref node
 * for the aggregate as partial.
 */
static bool
check_for_partialize_function_call(Node *node, PartializeWalkerState *state)
{
	if (node == NULL)
		return false;

	/*
	 * If the last node we saw was partialize, the next one must be aggregate
	 * we're partializing
	 */
	if (state->looking_for_agg && !IsA(node, Aggref))
		elog(ERROR, "the input to partialize must be an aggregate");

	if (IsA(node, Aggref))
	{
		Aggref *aggref = castNode(Aggref, node);

		if (state->looking_for_agg)
		{
			state->looking_for_agg = false;

			if (state->fix_aggref != TS_DO_NOT_FIX_AGGSPLIT)
			{
				if (state->fix_aggref == TS_FIX_AGGSPLIT_SIMPLE &&
					aggref->aggsplit == AGGSPLIT_SIMPLE)
				{
					aggref->aggsplit = AGGSPLIT_INITIAL_SERIAL;
				}
				else if (state->fix_aggref == TS_FIX_AGGSPLIT_FINAL &&
						 aggref->aggsplit == AGGSPLIT_FINAL_DESERIAL)
				{
					aggref->aggsplit = AGGSPLITOP_COMBINE | AGGSPLITOP_DESERIALIZE |
									   AGGSPLITOP_SERIALIZE | AGGSPLITOP_SKIPFINAL;
				}

				if (aggref->aggtranstype == INTERNALOID)
					aggref->aggtype = BYTEAOID;
				else
					aggref->aggtype = aggref->aggtranstype;
			}
		}

		/* We currently cannot handle cases like
		 *     SELECT sum(i), partialize(sum(i)) ...
		 *
		 * We check for non-partial aggs to ensure that if any of the aggregates
		 * in a statement are partialized, all of them have to be.
		 */
		else if (aggref->aggsplit != AGGSPLIT_INITIAL_SERIAL)
			state->found_non_partial_agg = true;
	}
	else if (IsA(node, FuncExpr) && ((FuncExpr *) node)->funcid == state->fnoid)
	{
		state->found_partialize = true;
		state->looking_for_agg = true;
	}

	return expression_tree_walker(node, check_for_partialize_function_call, state);
}

bool
has_partialize_function(Node *node, PartializeAggFixAggref fix_aggref)
{
	Oid partialfnoid = InvalidOid;
	Oid argtyp[] = { ANYELEMENTOID };

	PartializeWalkerState state = { .found_partialize = false,
									.found_non_partial_agg = false,
									.looking_for_agg = false,
									.fix_aggref = fix_aggref,
									.fnoid = InvalidOid };
	List *name = list_make2(makeString(INTERNAL_SCHEMA_NAME), makeString(TS_PARTIALFN));

	partialfnoid = LookupFuncName(name, lengthof(argtyp), argtyp, false);
	Assert(OidIsValid(partialfnoid));
	state.fnoid = partialfnoid;
	check_for_partialize_function_call(node, &state);

	if (state.found_partialize && state.found_non_partial_agg)
		elog(ERROR, "cannot mix partialized and non-partialized aggregates in the same statement");

	return state.found_partialize;
}

/*
 * Modify all AggPaths in relation to use partial aggregation.
 *
 * Note that there can be both parallel (split) paths and non-parallel
 * (non-split) paths suggested at this stage, but all of them refer to the
 * same Aggrefs. Depending on the Path picked, the Aggrefs are "fixed up" by
 * the PostgreSQL planner at a later stage in planner (in setrefs.c) to match
 * the choice of Path. For this reason, it is not possible to modify Aggrefs
 * at this stage AND keep both type of Paths. Therefore, if a split Path is
 * found, then prune the non-split path.
 */
static bool
partialize_agg_paths(RelOptInfo *rel)
{
	ListCell *lc;
	bool has_combine = false;
	List *aggsplit_simple_paths = NIL;
	List *aggsplit_final_paths = NIL;
	List *other_paths = NIL;

	foreach (lc, rel->pathlist)
	{
		Path *path = lfirst(lc);

		if (IsA(path, AggPath))
		{
			AggPath *agg = castNode(AggPath, path);

			if (agg->aggsplit == AGGSPLIT_SIMPLE)
			{
				agg->aggsplit = AGGSPLIT_INITIAL_SERIAL;
				aggsplit_simple_paths = lappend(aggsplit_simple_paths, path);
			}
			else if (agg->aggsplit == AGGSPLIT_FINAL_DESERIAL)
			{
				has_combine = true;
				aggsplit_final_paths = lappend(aggsplit_final_paths, path);
			}
			else
			{
				other_paths = lappend(other_paths, path);
			}
		}
		else
		{
			other_paths = lappend(other_paths, path);
		}
	}

	if (aggsplit_final_paths != NIL)
		rel->pathlist = list_concat(other_paths, aggsplit_final_paths);
	else
		rel->pathlist = list_concat(other_paths, aggsplit_simple_paths);

	return has_combine;
}

/* Get an an existing aggregation path for the given relation or NULL if no aggregation path exists.
 */
static bool
has_min_max_agg_path(RelOptInfo *relation)
{
	ListCell *lc;
	foreach (lc, relation->pathlist)
	{
		Path *path = lfirst(lc);
		if (IsA(path, MinMaxAggPath))
			return true;
	}

	return false;
}

/* Get an an existing aggregation path for the given relation or NULL if no aggregation path exists.
 */
static AggPath *
get_existing_agg_path(RelOptInfo *relation)
{
	ListCell *lc;
	foreach (lc, relation->pathlist)
	{
		Path *path = lfirst(lc);
		if (IsA(path, AggPath))
		{
			AggPath *existing_agg_path = castNode(AggPath, path);
			return existing_agg_path;
		}
	}

	return NULL;
}

/* Get all subpaths from a Append, MergeAppend, or ChunkAppend path */
static List *
get_subpaths_from_append_path(Path *path, bool handle_gather_path)
{
	if (IsA(path, AppendPath))
	{
		AppendPath *append_path = castNode(AppendPath, path);
		return append_path->subpaths;
	}
	else if (IsA(path, MergeAppendPath))
	{
		MergeAppendPath *merge_append_path = castNode(MergeAppendPath, path);
		return merge_append_path->subpaths;
	}
	else if (ts_is_chunk_append_path(path))
	{
		CustomPath *custom_path = castNode(CustomPath, path);
		return custom_path->custom_paths;
	}
	else if (handle_gather_path && IsA(path, GatherPath))
	{
		return get_subpaths_from_append_path(castNode(GatherPath, path)->subpath, false);
	}

	/* Aggregation push-down is not supported for other path types so far */
	return NIL;
}

/* Copy an AppendPath and set new subpaths. */
static AppendPath *
copy_append_path(AppendPath *path, List *subpaths)
{
	AppendPath *newPath = palloc(sizeof(AppendPath));
	memcpy(newPath, path, sizeof(AppendPath));
	newPath->subpaths = subpaths;
	cost_append(newPath);

	return newPath;
}

/* Copy a MergeAppendPath and set new subpaths. */
static MergeAppendPath *
copy_merge_append_path(PlannerInfo *root, MergeAppendPath *path, List *subpaths)
{
	MergeAppendPath *newPath = create_merge_append_path_compat(root,
															   path->path.parent,
															   subpaths,
															   path->path.pathkeys,
															   NULL,
															   path->partitioned_rels);

#if PG14_LT
	newPath->partitioned_rels = list_copy(path->partitioned_rels);
#endif

	newPath->path.param_info = path->path.param_info;

	return newPath;
}

/*
 * Generate a total aggregation path for partial aggregations
 */
static void
generate_agg_pushdown_path(PlannerInfo *root, Path *cheapest_total_path, RelOptInfo *output_rel,
						   RelOptInfo *partially_grouped_rel, PathTarget *grouping_target,
						   PathTarget *partial_grouping_target, bool can_sort, bool can_hash,
						   double d_num_groups, GroupPathExtraData *extra_data)
{
	Query *parse = root->parse;

	/* Determine costs for aggregations */
	AggClauseCosts *agg_partial_costs = &extra_data->agg_partial_costs;

	/* Get subpaths */
	List *subpaths = get_subpaths_from_append_path(cheapest_total_path, false);

	/* No subpaths available or unsupported append node */
	if (subpaths == NIL)
		return;

	/* Replan aggregation path */
	output_rel->pathlist = NIL;

	/* Generate agg paths on top of the append children */
	ListCell *lc;
	List *sorted_subpaths = NIL;
	List *hashed_subpaths = NIL;

	foreach (lc, subpaths)
	{
		Path *subpath = lfirst(lc);

		/* Translate targetlist for partition */
		AppendRelInfo *appinfo = ts_get_appendrelinfo(root, subpath->parent->relid, false);
		PathTarget *mypartialtarget = copy_pathtarget(partial_grouping_target);
		mypartialtarget->exprs =
			castNode(List,
					 adjust_appendrel_attrs(root, (Node *) mypartialtarget->exprs, 1, &appinfo));

		/* Usually done by appy_scanjoin_target_to_path */
		Assert(list_length(subpath->pathtarget->exprs) ==
			   list_length(cheapest_total_path->pathtarget->exprs));
		subpath->pathtarget->sortgrouprefs = cheapest_total_path->pathtarget->sortgrouprefs;

		if (can_sort)
		{
			int presorted_keys;
			bool is_sorted = pathkeys_count_contained_in(root->group_pathkeys,
														 subpath->pathkeys,
														 &presorted_keys);

			/* Use a copy of the subpath because it might get modified (i.e., sorted) */
			Path *sorted_path = subpath;

			if (!is_sorted)
			{
				sorted_path = (Path *) create_sort_path(root,
														subpath->parent,
														sorted_path,
														root->group_pathkeys,
														-1.0);
			}

			Path *sorted_agg_path =
				(Path *) create_agg_path(root,
										 sorted_path->parent,
										 sorted_path,
										 mypartialtarget,
										 parse->groupClause ? AGG_SORTED : AGG_PLAIN,
										 AGGSPLIT_INITIAL_SERIAL,
										 parse->groupClause,
										 NIL,
										 agg_partial_costs,
										 d_num_groups);

			sorted_subpaths = lappend(sorted_subpaths, sorted_agg_path);
		}

		if (can_hash)
		{
			Path *hash_path = (Path *) create_agg_path(root,
													   subpath->parent,
													   subpath,
													   mypartialtarget,
													   AGG_HASHED,
													   AGGSPLIT_INITIAL_SERIAL,
													   parse->groupClause,
													   NIL,
													   agg_partial_costs,
													   d_num_groups);

			hashed_subpaths = lappend(hashed_subpaths, hash_path);
		}
	}

	/* Create new append paths */
	cheapest_total_path->pathtarget = partial_grouping_target;

	if (IsA(cheapest_total_path, AppendPath))
	{
		AppendPath *append_path = castNode(AppendPath, cheapest_total_path);
		if (sorted_subpaths != NIL)
		{
			add_path(partially_grouped_rel,
					 (Path *) copy_append_path(append_path, sorted_subpaths));
		}

		if (hashed_subpaths != NIL)
		{
			add_path(partially_grouped_rel,
					 (Path *) copy_append_path(append_path, hashed_subpaths));
		}
	}
	else if (IsA(cheapest_total_path, MergeAppendPath))
	{
		MergeAppendPath *merge_append_path = castNode(MergeAppendPath, cheapest_total_path);
		if (sorted_subpaths != NIL)
		{
			add_path(partially_grouped_rel,
					 (Path *) copy_merge_append_path(root, merge_append_path, sorted_subpaths));
		}

		if (hashed_subpaths != NIL)
		{
			add_path(partially_grouped_rel,
					 (Path *) copy_merge_append_path(root, merge_append_path, hashed_subpaths));
		}
	}
	else if (ts_is_chunk_append_path(cheapest_total_path))
	{
		CustomPath *custom_path = castNode(CustomPath, cheapest_total_path);
		ChunkAppendPath *chunk_append_path = (ChunkAppendPath *) custom_path;
		if (sorted_subpaths != NIL)
		{
			add_path(partially_grouped_rel,
					 (Path *) ts_chunk_append_path_copy(chunk_append_path, sorted_subpaths));
		}

		if (hashed_subpaths != NIL)
		{
			add_path(partially_grouped_rel,
					 (Path *) ts_chunk_append_path_copy(chunk_append_path, hashed_subpaths));
		}
	}
	else
	{
		/* Should never happen, already checked above */
		Ensure(false, "Unknown path type");
	}
}

/*
 * Generate a partial aggregation path for partial aggregations
 */
static void
generate_partial_agg_pushdown_path(PlannerInfo *root, Path *cheapest_partial_path,
								   RelOptInfo *output_rel, RelOptInfo *partially_grouped_rel,
								   PathTarget *grouping_target, PathTarget *partial_grouping_target,
								   bool can_sort, bool can_hash, double d_num_groups,
								   GroupPathExtraData *extra_data)
{
	Query *parse = root->parse;

	/* Determine costs for aggregations */
	AggClauseCosts *agg_partial_costs = &extra_data->agg_partial_costs;

	/* Get subpaths */
	List *subpaths = get_subpaths_from_append_path(cheapest_partial_path, false);

	/* No subpaths available or unsupported append node */
	if (subpaths == NIL)
		return;

	/* Replan aggregation path */
	output_rel->partial_pathlist = NIL;

	/* Generate agg paths on top of the append children */
	ListCell *lc;
	List *sorted_subpaths = NIL;
	List *hashed_subpaths = NIL;

	foreach (lc, subpaths)
	{
		Path *subpath = lfirst(lc);

		Assert(subpath->parallel_safe);

		/* Translate targetlist for partition */
		AppendRelInfo *appinfo = ts_get_appendrelinfo(root, subpath->parent->relid, false);
		PathTarget *mypartialtarget = copy_pathtarget(partial_grouping_target);
		mypartialtarget->exprs =
			castNode(List,
					 adjust_appendrel_attrs(root, (Node *) mypartialtarget->exprs, 1, &appinfo));

		/* Usually done by appy_scanjoin_target_to_path */
		Assert(list_length(subpath->pathtarget->exprs) ==
			   list_length(cheapest_partial_path->pathtarget->exprs));
		subpath->pathtarget->sortgrouprefs = cheapest_partial_path->pathtarget->sortgrouprefs;

		if (can_sort)
		{
			int presorted_keys;
			bool is_sorted = pathkeys_count_contained_in(root->group_pathkeys,
														 subpath->pathkeys,
														 &presorted_keys);

			/* Use a copy of the subpath because it might get modified (i.e., sorted) */
			Path *sorted_path = subpath;

			if (!is_sorted)
			{
				sorted_path = (Path *) create_sort_path(root,
														subpath->parent,
														sorted_path,
														root->group_pathkeys,
														-1.0);
			}

			Path *sorted_agg_path =
				(Path *) create_agg_path(root,
										 sorted_path->parent,
										 sorted_path,
										 mypartialtarget,
										 parse->groupClause ? AGG_SORTED : AGG_PLAIN,
										 AGGSPLIT_INITIAL_SERIAL,
										 parse->groupClause,
										 NIL,
										 agg_partial_costs,
										 d_num_groups);

			sorted_subpaths = lappend(sorted_subpaths, sorted_agg_path);
		}

		if (can_hash)
		{
			Path *hash_path = (Path *) create_agg_path(root,
													   subpath->parent,
													   subpath,
													   mypartialtarget,
													   AGG_HASHED,
													   AGGSPLIT_INITIAL_SERIAL,
													   parse->groupClause,
													   NIL,
													   agg_partial_costs,
													   d_num_groups);

			hashed_subpaths = lappend(hashed_subpaths, hash_path);
		}
	}

	/* Create new append paths */
	cheapest_partial_path->pathtarget = partial_grouping_target;
	partially_grouped_rel->partial_pathlist = NIL;

	if (IsA(cheapest_partial_path, AppendPath))
	{
		AppendPath *append_path = castNode(AppendPath, cheapest_partial_path);
		if (sorted_subpaths != NIL)
		{
			AppendPath *new_agg_path = copy_append_path(append_path, sorted_subpaths);
			add_partial_path(partially_grouped_rel, (Path *) new_agg_path);
		}

		if (hashed_subpaths != NIL)
		{
			AppendPath *new_agg_path = copy_append_path(append_path, hashed_subpaths);
			add_partial_path(partially_grouped_rel, (Path *) new_agg_path);
		}
	}
	else if (IsA(cheapest_partial_path, MergeAppendPath))
	{
		MergeAppendPath *merge_append_path = castNode(MergeAppendPath, cheapest_partial_path);
		if (sorted_subpaths != NIL)
		{
			MergeAppendPath *new_agg_path =
				copy_merge_append_path(root, merge_append_path, sorted_subpaths);
			add_partial_path(partially_grouped_rel, (Path *) new_agg_path);
		}

		if (hashed_subpaths != NIL)
		{
			MergeAppendPath *new_agg_path =
				copy_merge_append_path(root, merge_append_path, hashed_subpaths);
			add_partial_path(partially_grouped_rel, (Path *) new_agg_path);
		}
	}
	else if (ts_is_chunk_append_path(cheapest_partial_path))
	{
		CustomPath *custom_path = castNode(CustomPath, cheapest_partial_path);
		ChunkAppendPath *chunk_append_path = (ChunkAppendPath *) custom_path;
		if (sorted_subpaths != NIL)
		{
			ChunkAppendPath *new_agg_path =
				ts_chunk_append_path_copy(chunk_append_path, sorted_subpaths);
			add_partial_path(partially_grouped_rel, (Path *) new_agg_path);
		}

		if (hashed_subpaths != NIL)
		{
			ChunkAppendPath *new_agg_path =
				ts_chunk_append_path_copy(chunk_append_path, hashed_subpaths);
			add_partial_path(partially_grouped_rel, (Path *) new_agg_path);
		}
	}
	else
	{
		/* Should never happen, already checked above */
		Ensure(false, "Unable to create partial aggregates - unknown path type");
	}

	/* Finish partial paths by adding a gather node */
	foreach (lc, partially_grouped_rel->partial_pathlist)
	{
		Path *append_path = lfirst(lc);
		double total_groups = append_path->rows * append_path->parallel_workers;

		Path *gather_path = (Path *) create_gather_path(root,
														partially_grouped_rel,
														append_path,
														partially_grouped_rel->reltarget,
														NULL,
														&total_groups);
		add_path(partially_grouped_rel, (Path *) gather_path);
	}
}

/*
 * Convert the aggregation into a partial aggregation and push them down to the chunk level
 *
 * Inspired by PostgreSQL's create_partitionwise_grouping_paths() function
 *
 * Generated aggregation paths:
 *
 * Finalize Aggregate
 *   -> Append
 *      -> Partial Aggregation 1
 *        - Chunk 1
 *      ...
 *      -> Partial Aggregation N
 *        - Chunk N
 */
void
ts_pushdown_partial_agg(PlannerInfo *root, Hypertable *ht, RelOptInfo *input_rel,
						RelOptInfo *output_rel, void *extra)
{
	Query *parse = root->parse;

	/* We are only interested in hypertables */
	if (ht == NULL || hypertable_is_distributed(ht))
		return;

	/* Perform partial aggregation planning only if there is an aggregation is requested */
	if (!parse->hasAggs)
		return;

	/* We can only perform a partial partitionwise aggregation, if no grouping is performed */
	if (parse->groupingSets)
		return;

	/* Don't replan aggregation if we already have a MinMaxAggPath (e.g., created by ts_preprocess_first_last_aggregates) */
	if(has_min_max_agg_path(output_rel))
		return;

	bool can_sort = grouping_is_sortable(parse->groupClause);
	bool can_hash = grouping_is_hashable(parse->groupClause) &&
					!parse->groupingSets; /* see consider_groupingsets_paths */

	/* No sorted or hashed aggregation possible, nothing to do for us */
	if (!can_sort && !can_hash)
		return;

	Assert(extra != NULL);
	GroupPathExtraData *extra_data = (GroupPathExtraData *) extra;

/* Don't replan aggregation if it contains already partials or non-serializable aggregates */
#if PG14_LT
	if (extra_data->agg_partial_costs.hasNonPartial || extra_data->agg_partial_costs.hasNonSerial)
#else
	if (root->hasNonPartialAggs || root->hasNonSerialAggs)
#endif
		return;

	/* Determine the number of groups from the already planned aggregation */
	AggPath *existing_agg_path = get_existing_agg_path(output_rel);
	if (existing_agg_path == NULL)
		return;

	/* Skip partial aggregations created by _timescaledb_internal.partialize_agg */
	if (existing_agg_path->aggsplit == AGGSPLIT_INITIAL_SERIAL)
		return;

	double d_num_groups = existing_agg_path->numGroups;
	Assert(d_num_groups > 0);

	/* Construct partial group agg upper rel */
	RelOptInfo *partially_grouped_rel =
		fetch_upper_rel(root, UPPERREL_PARTIAL_GROUP_AGG, input_rel->relids);
	partially_grouped_rel->consider_parallel = input_rel->consider_parallel;
	partially_grouped_rel->reloptkind = input_rel->reloptkind;
	partially_grouped_rel->serverid = input_rel->serverid;
	partially_grouped_rel->userid = input_rel->userid;
	partially_grouped_rel->useridiscurrent = input_rel->useridiscurrent;
	partially_grouped_rel->fdwroutine = input_rel->fdwroutine;

	/* Build target list for partial aggregate paths */
	PathTarget *grouping_target = output_rel->reltarget;
	PathTarget *partial_grouping_target = ts_make_partial_grouping_target(root, grouping_target);
	partially_grouped_rel->reltarget = partial_grouping_target;

	/* Generate the aggregation pushdown path */
	Path *cheapest_total_path = input_rel->cheapest_total_path;
	Assert(cheapest_total_path != NULL);
	generate_agg_pushdown_path(root,
							   cheapest_total_path,
							   output_rel,
							   partially_grouped_rel,
							   grouping_target,
							   partial_grouping_target,
							   can_sort,
							   can_hash,
							   d_num_groups,
							   extra_data);

	/* The same as above but for partial paths */
	if (input_rel->partial_pathlist != NIL && input_rel->consider_parallel)
	{
		Path *cheapest_partial_path = linitial(input_rel->partial_pathlist);
		generate_partial_agg_pushdown_path(root,
										   cheapest_partial_path,
										   output_rel,
										   partially_grouped_rel,
										   grouping_target,
										   partial_grouping_target,
										   can_sort,
										   can_hash,
										   d_num_groups,
										   extra_data);
	}

	/* Replan if we were able to generate partially grouped rel paths */
	if (partially_grouped_rel->pathlist == NIL)
		return;

	/* Prefer our paths */
	output_rel->pathlist = NIL;
	output_rel->partial_pathlist = NIL;

	/* Finalize partially aggregated append paths */
	AggClauseCosts *agg_final_costs = &extra_data->agg_final_costs;
	ListCell *lc;
	foreach (lc, partially_grouped_rel->pathlist)
	{
		Path *append_path = lfirst(lc);
		List *subpaths = get_subpaths_from_append_path(append_path, true);
		Assert(subpaths != NIL);

		AggPath *agg_path = castNode(AggPath, linitial(subpaths));

		if (agg_path->aggstrategy != AGG_HASHED)
		{
			bool is_sorted;
			int presorted_keys;

			is_sorted = pathkeys_count_contained_in(root->group_pathkeys,
													append_path->pathkeys,
													&presorted_keys);

			if (!is_sorted)
			{
				append_path = (Path *)
					create_sort_path(root, output_rel, append_path, root->group_pathkeys, -1.0);
			}

			add_path(output_rel,
					 (Path *) create_agg_path(root,
											  output_rel,
											  append_path,
											  grouping_target,
											  parse->groupClause ? AGG_SORTED : AGG_PLAIN,
											  AGGSPLIT_FINAL_DESERIAL,
											  parse->groupClause,
											  (List *) parse->havingQual,
											  agg_final_costs,
											  d_num_groups));
		}
		else
		{
			add_path(output_rel,
					 (Path *) create_agg_path(root,
											  output_rel,
											  append_path,
											  grouping_target,
											  AGG_HASHED,
											  AGGSPLIT_FINAL_DESERIAL,
											  parse->groupClause,
											  (List *) parse->havingQual,
											  agg_final_costs,
											  d_num_groups));
		}
	}
}

/*
 * Turn an aggregate relation into a partial aggregate relation if aggregates
 * are enclosed by the partialize_agg function.
 *
 * The partialize_agg function can "manually" partialize an aggregate. For
 * instance:
 *
 *  SELECT time_bucket('1 day', time), device,
 *  _timescaledb_internal.partialize_agg(avg(temp))
 *  GROUP BY 1, 2;
 *
 * Would compute the partial aggregate of avg(temp).
 *
 * The plan to compute the relation must be either entirely non-partial or
 * entirely partial, so it is not possible to mix partials and
 * non-partials. Note that aggregates can appear in both the target list and the
 * HAVING clause, for instance:
 *
 *  SELECT time_bucket('1 day', time), device, avg(temp)
 *  GROUP BY 1, 2
 *  HAVING avg(temp) > 3;
 *
 * Regular partial aggregations executed by the planner (i.e., those not induced
 * by the partialize_agg function) have their HAVING aggregates transparently
 * moved to the target list during planning so that the finalize node can use it
 * when applying the final filter on the resulting groups, obviously omitting
 * the extra columns in the final output/projection. However, it doesn't make
 * much sense to transparently do that when partializing with partialize_agg
 * since it would be odd to return more columns than requested by the
 * user. Therefore, the caller would have to do that manually. This, in fact, is
 * also done when materializing continuous aggregates.
 *
 * For this reason, HAVING clauses with partialize_agg are blocked, except in
 * cases where the planner transparently reduces the having expression to a
 * simple filter (e.g., HAVING device > 3). In such cases, the HAVING clause is
 * removed and replaced by a filter on the input.
 * Returns : true if partial aggs were found, false otherwise.
 * Modifies : output_rel if partials aggs were found.
 */
bool
ts_plan_process_partialize_agg(PlannerInfo *root, RelOptInfo *output_rel)
{
	Query *parse = root->parse;
	bool found_partialize_agg_func;

	Assert(IS_UPPER_REL(output_rel));

	if (CMD_SELECT != parse->commandType || !parse->hasAggs)
		return false;

	found_partialize_agg_func =
		has_partialize_function((Node *) parse->targetList, TS_DO_NOT_FIX_AGGSPLIT);

	if (!found_partialize_agg_func)
		return false;

	/* partialize_agg() function found. Now turn simple (non-partial) aggs
	 * (AGGSPLIT_SIMPLE) into partials. If the Agg is a combine/final we want
	 * to do the combine but not the final step. However, it is not possible
	 * to change that here at the Path stage because the PostgreSQL planner
	 * will hit an assertion, so we defer that to the plan stage in planner.c.
	 */
	bool is_combine = partialize_agg_paths(output_rel);

	if (!is_combine)
		has_partialize_function((Node *) parse->targetList, TS_FIX_AGGSPLIT_SIMPLE);

	/* We cannot check root->hasHavingqual here because sometimes the
	 * planner can replace the HAVING clause with a simple filter. But
	 * root->hashavingqual stays true to remember that the query had a
	 * HAVING clause initially. */
	if (NULL != parse->havingQual)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cannot partialize aggregate with HAVING clause"),
				 errhint("Any aggregates in a HAVING clause need to be partialized in the output "
						 "target list.")));

	return true;
}
