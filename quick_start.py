import subprocess
import sys
import logging
import concurrent.futures

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_script(script_path):
    """Runs a python script as a subprocess."""
    logging.info(f"Started: {script_path}")
    try:
        result = subprocess.run(
            [sys.executable, script_path], check=True, capture_output=True, text=True
        )
        logging.info(f"Success: {script_path}")
        return script_path, True, result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in {script_path}:\n{e.stderr}")
        return script_path, False, e.stderr


def run_in_parallel(scripts, max_workers=None):
    """Runs a list of scripts in parallel and waits for all to finish."""
    if not scripts:
        return True

    logging.info(f"Running {len(scripts)} scripts in parallel...")
    all_successful = True

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_script, script): script for script in scripts}
        for future in concurrent.futures.as_completed(futures):
            script_path = futures[future]
            try:
                _, success, _ = future.result()
                if not success:
                    all_successful = False
            except Exception as exc:
                logging.error(f"{script_path} generated an exception: {exc}")
                all_successful = False

    return all_successful


def run_sequentially(scripts):
    """Runs a list of scripts sequentially."""
    for script in scripts:
        _, success, _ = run_script(script)
        if not success:
            return False
    return True


def main():
    data_generation_scripts = [
        "scripts/generate_mc_data_static_skewed_t.py",
        "scripts/generate_mc_data_static_normal.py",
        "scripts/generate_mc_data_static_student_t.py",
        "scripts/generate_mc_data_dynamic_skewed_t.py",
        "scripts/generate_mc_data_historical_bootstrap.py",
        "scripts/generate_mc_data_static_normal_hmm.py",
        "scripts/generate_mc_data_static_skewed_t_hmm.py",
        "scripts/generate_mc_data_static_student_t_hmm.py",
        "scripts/generate_mc_data_dynamic_skewed_t_hmm.py",
    ]

    hmm_training_scripts = ["scripts/train_hmm_model.py"]

    backtest_scripts = [
        "scripts/generate_mc_backtests_static_skewed_t.py",
        "scripts/generate_mc_backtests_rHMM_MVO_static_skewed_t.py",
        "scripts/generate_mc_backtests_online_static_skewed_t.py",
        "scripts/generate_mc_backtests_online_rHMM_MVO_static_skewed_t.py",
    ]

    oos_prep_scripts = [
        "scripts/generate_mc_data_oos_static_skewed_t.py",
        "scripts/train_hmm_model_oos.py",
    ]

    oos_backtest_scripts = [
        "scripts/generate_mc_backtests_oos_static_skewed_t.py",
        "scripts/generate_mc_backtests_online_oos_static_skewed_t.py",
    ]

    logging.info("Starting Local Quickstart Pipeline Execution...")

    logging.info("--- PHASE 1: Data Generation ---")
    if not run_in_parallel(data_generation_scripts):
        logging.error("Phase 1 failed. Aborting pipeline.")
        sys.exit(1)

    logging.info("--- PHASE 2: HMM Training ---")
    if not run_sequentially(hmm_training_scripts):
        logging.error("Phase 2 failed. Aborting pipeline.")
        sys.exit(1)

    logging.info("--- PHASE 3: Backtests ---")
    if not run_in_parallel(backtest_scripts):
        logging.error("Phase 3 failed. Aborting pipeline.")
        sys.exit(1)

    logging.info("--- PHASE 4: OOS Data & HMM ---")
    if not run_sequentially(oos_prep_scripts):
        logging.error("Phase 4 failed. Aborting pipeline.")
        sys.exit(1)

    logging.info("--- PHASE 5: OOS Backtests ---")
    if not run_in_parallel(oos_backtest_scripts):
        logging.error("Phase 5 failed. Aborting pipeline.")
        sys.exit(1)

    logging.info("All pipeline phases completed successfully!")


if __name__ == "__main__":
    main()
