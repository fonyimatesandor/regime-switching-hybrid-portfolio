import pandas as pd
import pickle as pkl

from src.hmm.hmm_model import HMMModel
from src.hmm.feature_extraction import HMMFeatureExtractor
from src.utils.config_loader import load_config

config = load_config("config.yaml")

asset_data_learning = pd.read_csv(
    "data/raw/stock_data_learning.csv", index_col=0, parse_dates=True
)

asset_data_comparison = pd.read_csv(
    "data/raw/stock_data_comparison.csv", index_col=0, parse_dates=True
)

asset_data = pd.concat([asset_data_learning, asset_data_comparison], axis=0)


feature_extractor = HMMFeatureExtractor(
    vol_window=config["hmm_model"]["vol_window"],
    corr_window=config["hmm_model"]["corr_window"],
    draw_window=config["hmm_model"]["draw_window"],
)

X_features = feature_extractor.extract_features(asset_data)

hmm_model = HMMModel(
    n_components=config["hmm_model"]["n_components"],
    n_iter=config["hmm_model"]["n_iter"],
    df_bounds=tuple(config["hmm_model"]["df_bounds"]),
)

print("Training HMM model...")
hmm_model.fit(X_features)
print("HMM model trained successfully.")

with open("data/hmm_model/hmm_model_oos.pkl", "wb") as f:
    pkl.dump(hmm_model, f)

with open("data/hmm_model/hmm_feature_extractor_oos.pkl", "wb") as f:
    pkl.dump(feature_extractor, f)
