import argparse
import os
import torch
import pandas as pd
from src.utils.paths import get_dataset_paths, get_model_root, get_results_root
from src.utils.config import load_config, get_default_config
from src.data.dataset import build_splits, DopplerDataset1D, DopplerDataset2D
from src.models.registry import get_model
from src.training.trainer import train_model, load_ckpt
from src.utils.inference import run_inference
from src.utils.plotting import plot_results

def main():
    parser = argparse.ArgumentParser(description="DopplerLab CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "predict"], help="Mode to run")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, help="Model architecture name (e.g. cnn_1d, attn_2d)")
    parser.add_argument("--dataset-v", type=str, choices=["v1", "v2"], help="Dataset version override")
    parser.add_argument("--epochs", type=int, help="Epochs override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # 1. Load config
    config = get_default_config()
    if args.config:
        file_config = load_config(args.config)
        # Merge metrics
        if "metrics" in file_config:
            config.update(file_config["metrics"])
    
    # Overrides
    if args.epochs: config["epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    
    dataset_v = args.dataset_v if args.dataset_v else (file_config.get("dataset", {}).get("version", "v2") if args.config else "v2")
    model_name = args.model_name if args.model_name else (file_config.get("model", {}).get("name", "cnn_1d") if args.config else "cnn_1d")
    
    paths = get_dataset_paths(dataset_v)
    model_root = get_model_root()
    results_root = get_results_root()
    
    # 2. Setup Data
    train_files, val_files, test_files = build_splits(
        paths["audio_clips"], 
        n_clips=config["train_clips"],
        train_r=config["train_ratio"],
        val_r=config["val_ratio"],
        seed=config["seed"]
    )
    
    DatasetCls = DopplerDataset1D if "1d" in model_name else DopplerDataset2D
    
    # 3. Model
    model = get_model(model_name)
    ckpt_path = os.path.join(model_root, f"{model_name}_{dataset_v}.pt")
    
    if args.mode == "train":
        train_model(
            model_name=f"{model_name}-{dataset_v}",
            model=model,
            train_files=train_files,
            val_files=val_files,
            DatasetCls=DatasetCls,
            ckpt_path=ckpt_path,
            config=config,
            device=args.device
        )
    
    elif args.mode == "evaluate":
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return
            
        load_ckpt(ckpt_path, model, device=args.device)
        rows = run_inference(
            model_name=model_name,
            model=model,
            test_files=test_files,
            DatasetCls=DatasetCls,
            device=args.device,
            max_dist=paths["max_dist"]
        )
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(results_root, f"{model_name}_{dataset_v}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        fig_dir = os.path.join(results_root, "figures", f"{model_name}_{dataset_v}")
        plot_results(
            df=df,
            model_names=[model_name],
            colors={model_name: "coral" if "2d" in model_name else "steelblue"},
            fig_dir=fig_dir,
            speed_lim=[0, 55],
            dist_lim=[0, paths["max_dist"] + 10],
            dist_bins=paths["dist_bins"],
            bin_labels=paths["bin_labels"],
            max_dist=paths["max_dist"]
        )
        print(f"Plots saved to {fig_dir}")

if __name__ == "__main__":
    main()
