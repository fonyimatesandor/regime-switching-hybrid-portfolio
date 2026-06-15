import json
import glob
import os

def main():
    json_files = glob.glob("data/mc_metrics/mc_backtest_summary_*.json")
    
    if not json_files:
        print("No separate summary JSON files found in data/mc_metrics/ matching 'mc_backtest_summary_*.json'")
        return
        
    master_models_tested = []
    master_simulators_tested = []
    master_config = None
    
    for f in json_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                
                # Maintain order while appending unique models
                for model in data.get("models_tested", []):
                    if model not in master_models_tested:
                        master_models_tested.append(model)
                        
                # Maintain order while appending unique simulators
                for sim in data.get("simulators_tested", []):
                    if sim not in master_simulators_tested:
                        master_simulators_tested.append(sim)
                        
                if master_config is None and "config" in data:
                    master_config = data["config"]
                    
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    master_data = {
        "models_tested": master_models_tested,
        "simulators_tested": master_simulators_tested,
        "config": master_config
    }
    
    output_file = "data/mc_metrics/mc_backtest_summary.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as out_f:
        json.dump(master_data, out_f, indent=4)
        
    print(f"Successfully merged {len(json_files)} JSON files into {output_file}")
    print(f"Total models tested: {len(master_models_tested)}")
    print(f"Total simulators tested: {len(master_simulators_tested)}")

if __name__ == "__main__":
    main()
