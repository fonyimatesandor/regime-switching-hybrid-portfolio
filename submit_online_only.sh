#!/bin/bash

echo "Starting Online Backtests Submission..."

O1_ID=$(qsub generate_mc_backtests_online_historical_bootstrap.pbs)
O2_ID=$(qsub generate_mc_backtests_online_static_normal.pbs)
O3_ID=$(qsub generate_mc_backtests_online_static_skewed_t.pbs)
O4_ID=$(qsub generate_mc_backtests_online_static_student_t.pbs)
O5_ID=$(qsub generate_mc_backtests_online_dynamic_skewed_t.pbs)

echo "1. Submitted Online Backtests: $O1_ID, $O2_ID, $O3_ID, $O4_ID, $O5_ID"

MERGE_ID=$(qsub -W depend=afterok:${O1_ID}:${O2_ID}:${O3_ID}:${O4_ID}:${O5_ID} merge_mc_summary_jsons.pbs)
echo "2. Submitted JSON Merger: $MERGE_ID"

echo "Online jobs and merger successfully submitted to PBS!"