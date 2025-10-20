"""
Inference script for chess move prediction.
Loads trained models and generates predictions.
"""

import os
import json
import torch
import shutil
from torch.utils.data import DataLoader
from dataset import move_to_idx, ChessFENDataset, idx_to_move
from model import MODELS_DICT


def inference(
    model_path: str,
    input_dim: int,
    num_classes: int,
    dataloader: DataLoader,
    rules: str,
    games_nb: int,
    output_path: str,
    device: torch.device
):
    """
    Run inference and save predictions.
    
    Args:
        model_path: Path to saved model checkpoint
        input_dim: Input feature dimension
        num_classes: Number of output classes (possible moves)
        dataloader: DataLoader for validation set
        rules: Rule enforcement mode ("withrules", "norules", "medium")
        games_nb: Number of training games (for naming)
        output_path: Path to save predictions
        device: Device to run inference on
    """
    key_name = f"{games_nb}_{rules}"
    print(f"Running inference for {key_name}...")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint['training_cfg'].model_name
    model = MODELS_DICT[model_name](input_dim=input_dim, num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load existing predictions file
    with open(output_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    # Index entries by ID for fast lookup
    entry_map = {entry["id"]: entry for entry in entries}

    # Generate predictions
    with torch.no_grad():
        for X, y, lm, data_ref, ids in dataloader:
            X = X.to(device).float()
            lm = lm.to(device)
            
            # Forward pass
            output = model(X)
            
            # Apply legal moves mask if using rules
            if rules in ["withrules", "medium"]:
                output = output + torch.log(lm.detach())

            # Get predicted move indices
            preds = torch.argmax(output, dim=1).cpu().tolist()

            # Save predictions to entries
            for i, sample_id in enumerate(ids):
                entry = entry_map.get(sample_id)
                if entry is None:
                    continue

                # Initialize model section if it doesn't exist
                if model_name not in entry:
                    entry[model_name] = {}

                # Save prediction as UCI move string
                entry[model_name][key_name] = idx_to_move[preds[i]]

    # Write updated entries back to file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Saved predictions for {key_name}")


if __name__ == "__main__":
    from config import Config

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    cfg = Config.from_json("configs/inference_config.json")

    # Create validation dataset and dataloader
    val_dataset = ChessFENDataset(cfg.data_path, move_to_idx, cfg.valgames_nb)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Get dimensions
    x, y, lm, _, _ = val_dataset[0]
    input_dim = x.shape[0]
    num_classes = lm.shape[0]

    # Prepare output file
    output_path = os.path.join(
        "experiments",
        cfg.expe,
        f"{cfg.ref_name}_inference.jsonl"
    )

    # Copy original data if output doesn't exist
    if not os.path.exists(output_path):
        shutil.copy(cfg.data_path, output_path)
        print(f"Created inference output file: {output_path}")

    # Run inference for all model configurations
    for games_nb in cfg.L_games_nb:
        for rules in cfg.L_rules:
            model_path = os.path.join(
                "experiments",
                cfg.expe,
                "models",
                f"best_model_{games_nb}_{rules}.pth"
            )
            
            # Run inference
            if rules == "norules":
                # For norules, also evaluate with medium mode
                inference(
                    model_path, input_dim, num_classes,
                    val_loader, rules, games_nb, output_path, device
                )
                inference(
                    model_path, input_dim, num_classes,
                    val_loader, "medium", games_nb, output_path, device
                )
            else:
                inference(
                    model_path, input_dim, num_classes,
                    val_loader, rules, games_nb, output_path, device
                )
    
    print("\n✓ Inference completed successfully!")