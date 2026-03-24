import os
from dotenv import load_dotenv

load_dotenv()

def get_data_root():
    return os.getenv("DATA_ROOT", "d:/Antigravity/DopplerLab/Datasets")

def get_dataset_version():
    return os.getenv("DATASET_VERSION", "v2")

def get_model_root():
    return os.getenv("MODEL_ROOT", "d:/Antigravity/DopplerLab/models")

def get_results_root():
    return os.getenv("RESULTS_ROOT", "d:/Antigravity/DopplerLab/results")

def get_log_root():
    return os.getenv("LOG_ROOT", "d:/Antigravity/DopplerLab/logs")

def get_dataset_paths(version=None):
    if version is None:
        version = get_dataset_version()
    
    data_root = get_data_root()
    
    if version == "v1":
        path = os.path.join(data_root, "neurips_v1", "audio_clips")
        max_dist = 120.0
        dist_bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 130)]
        bin_labels = ["0-20", "20-40", "40-60", "60-100", "100-130"]
    else:  # v2
        path = os.path.join(data_root, "neurips_v2", "audio_clips")
        max_dist = 1000.0
        dist_bins = [(0, 100), (100, 250), (250, 500), (500, 750), (750, 1010)]
        bin_labels = ["0-100", "100-250", "250-500", "500-750", "750-1000"]
        
    return {
        "audio_clips": path,
        "max_dist": max_dist,
        "dist_bins": dist_bins,
        "bin_labels": bin_labels
    }
