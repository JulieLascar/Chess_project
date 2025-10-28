"""
Training script for chess move prediction models.

Experiments with:
- Various training dataset sizes (1, 10, 100, 1000, 10000, 100000 games)
- Rules enforcement (with/without legal moves filtering)
- TensorBoard visualization for training monitoring

Training Modes:
- "norules": Model predicts any move (may be illegal)
- "withrules": Legal moves mask applied during training
- "medium": Model trained without rules, evaluated with rules
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from dataset import move_to_idx, ChessFENDataset
from model import MODELS_DICT

# Load configuration
cfg = Config.from_json("configs/train_config.json")

# Batch size mapping based on dataset size
# Smaller datasets use smaller batches for more gradient updates
BATCH_SIZE_MAP = {1: 16, 10: 64, 100: 128, 1000: 512, 10000: 1024, 100000: 1024}

VAL_BATCH_SIZE = BATCH_SIZE_MAP[cfg.valgames_nb]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize TensorBoard writer if enabled
if cfg.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"experiments/{cfg.expe}/runs/")


def train_one_epoch(model, dataloader, optimizer, loss_function, device, rules, learning_rate=None):
    """
    Train the model for one epoch.

    Args:
        model: Neural network model to train
        dataloader: Training data loader
        optimizer: Optimization algorithm (e.g., Adam)
        loss_function: Loss criterion (e.g., CrossEntropyLoss)
        device: Device to run training on (CPU or CUDA)
        rules: Rule enforcement mode ("withrules", "norules", "medium")

    Returns:
        Tuple of (average_loss, accuracy)

    Note:
        - "withrules" mode applies legal moves mask during training
        - This forces the model to only consider legal moves
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y, lm, data_ref, sample_id in dataloader:
        # Move data to device
        X = X.to(device).float()
        y = y.to(device).long()
        lm = lm.to(device)

        if optimizer is not None:
            optimizer.zero_grad()
        else:
            model.zero_grad()

        # Forward pass
        output = model(X)  # [batch_size, num_classes]

        # Apply legal moves mask if using rules during training
        if rules == "withrules":
            output = output + torch.log(lm.detach())

        loss = loss_function(output, y)

        loss.backward()

        if optimizer is not None:
            optimizer.step()
        else:
            # Manual gradient descent: w = w - lr * grad
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad

        # Calculate metrics
        with torch.no_grad():
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, loss_function, device, rules):
    """
    Evaluate the model on validation set.

    Args:
        model: Neural network model to evaluate
        dataloader: Validation data loader
        loss_function: Loss criterion
        device: Device to run evaluation on
        rules: Rule enforcement mode ("withrules", "norules", "medium")

    Returns:
        Tuple of (average_loss, accuracy)

    Note:
        - "medium" and "withrules" modes apply legal moves mask during evaluation
        - This ensures predictions are always legal moves
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y, lm, data_ref, sample_id in dataloader:
            # Move data to device
            X = X.to(device).float()
            y = y.to(device).long()
            lm = lm.to(device)

            # Forward pass
            output = model(X)  # [batch_size, num_classes]

            # Apply legal moves mask for withrules and medium modes
            if rules in ["withrules", "medium"]:
                output = output + torch.log(lm.detach())

            loss = loss_function(output, y)

            # Calculate metrics
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Main training loop: iterate over different dataset sizes
print("\n" + "=" * 70)
print("CHESS MOVE PREDICTION - TRAINING")
print("=" * 70)

for games_nb in cfg.L_games_nb:
    batch_size = BATCH_SIZE_MAP[games_nb]

    print(f"\n{'=' * 70}")
    print(f"Training with {games_nb} games")
    print(f"{'=' * 70}")

    # Create datasets and dataloaders
    train_dataset = ChessFENDataset(cfg.train_data_path, move_to_idx, games_nb)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ChessFENDataset(cfg.val_data_path, move_to_idx, cfg.valgames_nb)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # Get dimensions from first sample
    x, y, lm, _, _ = train_dataset[0]
    input_dim = x.shape[0]
    num_classes = lm.shape[0]

    print(f"Training dataset size: {len(train_dataset)} positions")
    print(f"Validation dataset size: {len(val_dataset)} positions")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes (possible moves): {num_classes}")
    print(f"Batch size: {batch_size}")

    # Train with different rule configurations
    for rules in cfg.L_rules:
        print(f"\n{'-' * 70}")
        print(f"Training mode: {rules}")
        print(f"{'-' * 70}")

        # Initialize model
        torch.manual_seed(42)  # For reproducibility
        model = MODELS_DICT[cfg.model_name](input_dim=input_dim, num_classes=num_classes)
        model.to(device)

        # Setup training
        loss_function = nn.CrossEntropyLoss()
        if cfg.use_optimizer:
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
            print(f"Using Adam optimizer with learning rate: {cfg.lr}")
        else:
            optimizer = None
            print(f"Using manual gradient descent with learning rate: {cfg.lr}")

        # Model saving setup
        best_val_acc = 0.0
        model_path = os.path.join("experiments", cfg.expe, "models")
        os.makedirs(model_path, exist_ok=True)
        best_model_path = os.path.join(model_path, f"best_model_{games_nb}_{rules}.pth")

        # Initial validation (before training)
        val_loss, val_acc = evaluate(model, val_loader, loss_function, device, rules)
        print(f"Initial validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if cfg.tensorboard:
            writer.add_scalars("Val-Loss", {f"{games_nb}_{rules}": val_loss}, 0)
            writer.add_scalars("Val-Accuracy", {f"{games_nb}_{rules}": val_acc}, 0)

            # Also log medium mode for norules
            if rules == "norules":
                val_loss_med, val_acc_med = evaluate(model, val_loader, loss_function, device, "medium")
                writer.add_scalars("Val-Loss", {f"{games_nb}_medium": val_loss_med}, 0)
                writer.add_scalars("Val-Accuracy", {f"{games_nb}_medium": val_acc_med}, 0)

        # Training loop
        print(f"\nStarting training for {cfg.epochs} epochs...")
        for epoch in range(cfg.epochs):
            # Train for one epoch
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_function, device, rules, cfg.lr)

            # Evaluate on validation set
            val_loss, val_acc = evaluate(model, val_loader, loss_function, device, rules)

            # Evaluate medium mode for norules (model without rules, eval with rules)
            if rules == "norules":
                val_loss_med, val_acc_med = evaluate(model, val_loader, loss_function, device, "medium")
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                    f"Val Loss (med): {val_loss_med:.4f}, Acc (med): {val_acc_med:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

            # Log to TensorBoard
            if cfg.tensorboard:
                # writer.add_scalars("Train-Loss", {f"{games_nb}_{rules}": train_loss}, epoch + 1)
                # writer.add_scalars("Train-Accuracy", {f"{games_nb}_{rules}": train_acc}, epoch + 1)
                writer.add_scalars("Val-Loss", {f"{games_nb}_{rules}": val_loss}, epoch + 1)
                writer.add_scalars("Val-Accuracy", {f"{games_nb}_{rules}": val_acc}, epoch + 1)

                if rules == "norules":
                    writer.add_scalars("Val-Loss", {f"{games_nb}_medium": val_loss_med}, epoch + 1)
                    writer.add_scalars("Val-Accuracy", {f"{games_nb}_medium": val_acc_med}, epoch + 1)

            # Save best model
            if cfg.save_model and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "training_cfg": cfg,
                    },
                    best_model_path,
                )
                print(f"    → New best model saved! (acc={val_acc:.4f})")

        print(f"\nTraining completed for {rules} mode.")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved to: {best_model_path}")

# Cleanup
if cfg.tensorboard:
    writer.close()
    print(f"\nTensorBoard logs saved to: experiments/{cfg.expe}/runs/")
    print(f"View with: tensorboard --logdir experiments/{cfg.expe}/runs/")

print("\n" + "=" * 70)
print("✓ ALL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
