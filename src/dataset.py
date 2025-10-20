"""
PyTorch Dataset for chess positions and moves.
Handles loading and encoding of FEN positions with legal moves.
"""

import json
import torch
from torch.utils.data import Dataset
from make_input import fen_to_feature_vector

# Generate all possible chess moves in UCI format
# Format: source_square + destination_square + optional_promotion
# Example: "e2e4", "e7e8q" (pawn promotion to queen)
ALL_MOVES = [
    a + str(b) + c + str(d) + e
    for a in "abcdefgh"
    for b in range(1, 9)
    for c in "abcdefgh"
    for d in range(1, 9)
    for e in ["", "r", "n", "b", "q"]  # Promotion pieces (empty for non-promotions)
]

# Bidirectional mapping between moves and indices
move_to_idx = {m: i for i, m in enumerate(ALL_MOVES)}
idx_to_move = {i: m for m, i in move_to_idx.items()}


class ChessFENDataset(Dataset):
    """
    PyTorch Dataset for chess positions in FEN format.
    
    Each sample contains:
        - Encoded FEN position (855-dimensional feature vector)
        - Target move (as class index)
        - Legal moves mask (binary vector)
        - Metadata (data reference and sample ID)
    
    Args:
        jsonl_path: Path to JSONL file containing chess positions
        move_to_idx: Dictionary mapping moves to integer indices
        games_nb: Number of games to load from the dataset
    
    Example JSONL format:
        {
            "id": "0_15",
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "legal_moves": ["e2e4", "d2d4", ...],
            "human": {"next_move": "e2e4"},
            "data_ref": "lichess-2025-07"
        }
    """

    def __init__(self, jsonl_path: str, move_to_idx: dict, games_nb: int):
        """
        Initialize the dataset by loading positions from JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            move_to_idx: Dictionary for move encoding
            games_nb: Maximum number of games to load
        """
        self.data = []
        self.move_to_idx = move_to_idx
        self.M = len(move_to_idx)  # Total number of possible moves (~20,480)

        # Load data from JSONL file, limiting to specified number of games
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                
                # Extract game number from ID (format: "game_move")
                game_id = int(sample["id"].split("_")[0])
                
                if game_id < games_nb:
                    self.data.append(sample)
                else:
                    break  # Stop once we've loaded enough games

    def __len__(self) -> int:
        """Return the total number of positions in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - x: Encoded FEN position tensor [855]
                - y: Target move index (int)
                - lm: Legal moves mask [M] (boolean tensor)
                - data_ref: Reference to source dataset (str)
                - sample_id: Sample identifier (str)
        """
        sample = self.data[idx]
        fen = sample["FEN"]
        next_move = sample["human"]["next_move"]
        legal_moves = sample["legal_moves"]
        data_ref = sample["data_ref"]
        sample_id = sample["id"]

        # Encode FEN position to feature vector [855]
        x = fen_to_feature_vector(fen)

        # Encode target move as integer label
        y = torch.tensor(self.move_to_idx[next_move], dtype=torch.long)

        # Create legal moves mask [M]
        # This is a boolean tensor where True indicates a legal move
        lm = torch.zeros(self.M, dtype=torch.bool)
        for move in legal_moves:
            if move in self.move_to_idx:
                lm[self.move_to_idx[move]] = True

        return x, y, lm, data_ref, sample_id


if __name__ == "__main__":
    # Test the dataset with sample data
    from torch.utils.data import DataLoader

    print("Testing ChessFENDataset...")
    
    # Create dataset with 5 games
    dataset = ChessFENDataset(
        "data/processed_data/lichess-2025-07.jsonl",
        move_to_idx,
        games_nb=5
    )
    
    print(f"Dataset size: {len(dataset)} positions")
    print(f"Total possible moves: {len(move_to_idx)}")
    
    # Test single sample
    print("\n--- Testing single sample ---")
    x, y, lm, data_ref, sample_id = dataset[0]
    print(f"Features shape: {x.shape}")           # [855]
    print(f"Features dtype: {x.dtype}")           # float32
    print(f"Label: {y.item()}")                   # integer
    print(f"Legal moves mask shape: {lm.shape}")  # [20480]
    print(f"Number of legal moves: {lm.sum().item()}")
    print(f"Data reference: {data_ref}")
    print(f"Sample ID: {sample_id}")
    
    # Test dataloader
    print("\n--- Testing DataLoader ---")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for x, y, lm, data_ref, sample_id in dataloader:
        print(f"Batch features shape: {x.shape}")      # [16, 855]
        print(f"Batch labels shape: {y.shape}")        # [16]
        print(f"Batch legal moves shape: {lm.shape}")  # [16, 20480]
        print(f"Batch data refs: {len(data_ref)}")     # 16
        print(f"Batch sample IDs: {len(sample_id)}")   # 16
        break
    
    print("\n✓ Dataset test completed successfully!")