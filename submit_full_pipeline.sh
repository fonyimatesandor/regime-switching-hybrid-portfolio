#!/bin/bash

echo "Starting Optimized Full MC Pipeline Submission..."

DATA_ID=$(qsub generate_mc_data.pbs)
echo "1. Submitted Data Generation: $DATA_ID"

echo "2. Submitting all Backtests (waiting on data generation)..."

BACKTEST_ID=$(qsub -W depend=afterok:$DATA_ID generate_mc_backtests.pbs)

O1_ID=$(qsub -W depend=afterok:$DATA_ID generate_mc_backtests_online_historical_bootstrap.pbs)
O2_ID=$(qsub -W depend=afterok:$DATA_ID generate_mc_backtests_online_static_normal.pbs)
O3_ID=$(qsub -W depend=afterok:$DATA_ID generate_mc_backtests_online_static_skewed_t.pbs)
O4_ID=$(qsub -W depend=afterok:$DATA_ID generate_mc_backtests_online_static_student_t.pbs)
O5_ID=$(qsub -W depend=afterok:$DATA_ID generate_mc_backtests_online_dynamic_skewed_t.pbs)

echo "   -> Standard Backtest queued: $BACKTEST_ID"
echo "   -> Online Backtests queued: $O1_ID, $O2_ID, $O3_ID, $O4_ID, $O5_ID"

MERGE_ID=$(qsub -W depend=afterok:${O1_ID}:${O2_ID}:${O3_ID}:${O4_ID}:${O5_ID} merge_mc_summary_jsons.pbs)
echo "3. Submitted JSON Merger: $MERGE_ID"

echo "Optimized pipeline successfully submitted to PBS!"