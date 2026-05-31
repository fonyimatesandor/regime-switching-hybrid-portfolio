import yaml
import numpy as np
from pathlib import Path

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file.
    """
    path = Path(config_path)
    
    if not path.exists():
        path = Path("../config.yaml")
        if not path.exists():
            raise FileNotFoundError("Could not locate config.yaml in current or parent directory.")
            
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        
    return config

def build_bounds_and_constraints(cfg):
    """
    Dynamically builds the bounds list and constraint dictionaries 
    for optimization based on the YAML config.
    """
    tickers = cfg['tickers']
    n_assets = len(tickers)
    
    default_min = cfg['asset_bounds']['default_min']
    default_max = cfg['asset_bounds']['default_max']
    
    bounds = [[default_min, default_max] for _ in range(n_assets)]
    
    for max_val, t_list in cfg['asset_bounds']['custom_max'].items():
        for t in t_list:
            if t in tickers:
                idx = tickers.index(t)
                bounds[idx][1] = float(max_val)
                
    bounds = [tuple(b) for b in bounds]

    static_constraints = []
    
    for sector_name, rules in cfg['sector_constraints'].items():
        sector_tickers = rules.get('tickers', [])
        idx_list = [tickers.index(t) for t in sector_tickers if t in tickers]
        
        if not idx_list:
            continue
            
        sector_idx = np.array(idx_list)
        
        if 'max_weight' in rules:
            max_w = float(rules['max_weight'])
            jac_max = np.zeros(n_assets)
            jac_max[sector_idx] = -1.0
            
            static_constraints.append({
                'type': 'ineq',
                'fun': lambda w, m=max_w, i=sector_idx: m - w[i].sum(),
                'jac': lambda w, j=jac_max: j
            })
            
        if 'min_weight' in rules:
            min_w = float(rules['min_weight'])
            jac_min = np.zeros(n_assets)
            jac_min[sector_idx] = 1.0
            
            static_constraints.append({
                'type': 'ineq',
                'fun': lambda w, m=min_w, i=sector_idx: w[i].sum() - m,
                'jac': lambda w, j=jac_min: j
            })

  
    dynamic_constraints = {}
  
    if 'dynamic_constraints' in cfg:
        
        dynamic_cfg = cfg['dynamic_constraints']
        
        if 'turnover_limit' in dynamic_cfg:
            dynamic_constraints['max_turnover'] = float(dynamic_cfg['turnover_limit']['max'])
            
        if 'rebalance_threshold' in dynamic_cfg:
            dynamic_constraints['min_diff_to_rebalance'] = float(dynamic_cfg['rebalance_threshold']['min'])

    return bounds, static_constraints, dynamic_constraints