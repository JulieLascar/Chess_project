"""
FEN (Forsyth-Edwards Notation) encoding module.
Converts chess positions from FEN format to PyTorch tensors for neural network input.

FEN Format Example:
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    Parts:
    1. Board position (piece placement)
    2. Active color (w/b)
    3. Castling availability (KQkq)
    4. En passant target square
    5. Halfmove clock (50-move rule)
    6. Fullmove number

Output Feature Vector:
    Total dimension: 855
    - Board: 832 (64 squares × 13 channels)
    - Turn: 1
    - Castling: 4
    - En passant: 16
    - Halfmove clock: 1
    - Fullmove number: 1
"""

import torch

# Piece mapping: uppercase = white, lowercase = black
# P=pawn, R=rook, N=knight, B=bishop, Q=queen, K=king
PIECES = ["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k"]
PIECE_TO_IDX = {p: i + 1 for i, p in enumerate(PIECES)}  # +1 because 0 = empty square
ONEHOTS = torch.eye(13)  # 13 = 12 pieces + 1 empty square

# En passant square mapping (only ranks 3 and 6 are valid for en passant)
EP_POSITIONS = [f"{file}{rank}" for rank in [3, 6] for file in "abcdefgh"]
EP_TO_IDX = {sq: i for i, sq in enumerate(EP_POSITIONS)}


def parse_board(board_fen: str) -> torch.Tensor:
    """
    Encode the board position from FEN string using one-hot encoding.
    
    The board is represented as 64 squares, each with 13 channels:
    - Channel 0: Empty square
    - Channels 1-6: White pieces (P, R, N, B, Q, K)
    - Channels 7-12: Black pieces (p, r, n, b, q, k)
    
    Args:
        board_fen: Board part of FEN notation (e.g., "rnbqkbnr/pppppppp/...")
                   Ranks are separated by '/', digits represent empty squares
        
    Returns:
        Tensor of shape [64, 13] representing the board state with one-hot encoding
        
    Example:
        >>> parse_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        # Returns [64, 13] tensor with starting position encoded
    """
    board = []
    for row in board_fen.split("/"):
        for char in row:
            if char.isdigit():  # Empty squares
                for _ in range(int(char)):
                    board.append(ONEHOTS[0])
            else:  # Piece
                board.append(ONEHOTS[PIECE_TO_IDX[char]])
    
    assert len(board) == 64, f"Board must have 64 squares, got {len(board)}"
    return torch.stack(board)


def parse_ep(ep: str) -> torch.Tensor:
    """
    Encode en passant target square.
    
    En passant is only possible on ranks 3 (for black) and 6 (for white),
    giving 16 possible squares (8 files × 2 ranks).
    
    Args:
        ep: En passant square in algebraic notation (e.g., "e3") or "-" if none
        
    Returns:
        Tensor of shape [16] with one-hot encoding of en passant square.
        All zeros if no en passant is possible.
        
    Example:
        >>> parse_ep("e3")
        # Returns [16] tensor with 1 at the index corresponding to e3
        >>> parse_ep("-")
        # Returns [16] tensor of all zeros
    """
    ep_tensor = torch.zeros(16)
    if ep in EP_TO_IDX:
        ep_tensor[EP_TO_IDX[ep]] = 1
    return ep_tensor


def parse_fen(fen: str) -> dict:
    """
    Parse complete FEN string into component tensors.
    
    Args:
        fen: Complete FEN string (e.g., "rnbqkbnr/pppppppp/... w KQkq - 0 1")
        
    Returns:
        Dictionary containing:
            - board: [64, 13] tensor for piece positions
            - turn: [1] tensor (1=white to move, 0=black to move)
            - castling: [4] tensor for castling rights (K, Q, k, q)
            - ep: [16] tensor for en passant square
            - fiftymove: [1] tensor for halfmove clock
            - nbmove: [1] tensor for fullmove number
            
    Example:
        >>> parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        {
            'board': torch.Tensor([64, 13]),
            'turn': torch.Tensor([1.0]),
            'castling': torch.Tensor([1., 1., 1., 1.]),
            'ep': torch.Tensor([0., 0., ..., 0.]),
            'fiftymove': torch.Tensor([0.]),
            'nbmove': torch.Tensor([1.])
        }
    """
    parts = fen.split()
    board_fen, turn, castling, ep, fiftymove, nbmove = parts[:6]

    # 1. Board position (64 squares × 13 channels)
    board = parse_board(board_fen)

    # 2. Turn (white=1, black=0)
    turn_tensor = torch.tensor([1 if turn == "w" else 0], dtype=torch.float32)

    # 3. Castling rights (K, Q, k, q)
    # K = White kingside, Q = White queenside
    # k = Black kingside, q = Black queenside
    castling_rights = torch.tensor(
        [
            1 if "K" in castling else 0,  # White kingside
            1 if "Q" in castling else 0,  # White queenside
            1 if "k" in castling else 0,  # Black kingside
            1 if "q" in castling else 0,  # Black queenside
        ],
        dtype=torch.float32,
    )

    # 4. En passant square (16 possible positions)
    ep_tensor = parse_ep(ep)

    # 5. Halfmove clock (for fifty-move rule)
    # Number of halfmoves since last capture or pawn move
    fiftymove_tensor = (
        torch.tensor([int(fiftymove)], dtype=torch.float32)
        if fiftymove
        else torch.tensor([0.0])
    )

    # 6. Fullmove number (starts at 1, increments after black's move)
    nbmove_tensor = (
        torch.tensor([int(nbmove)], dtype=torch.float32)
        if nbmove
        else torch.tensor([0.0])
    )

    return {
        "board": board,                    # [64, 13]
        "turn": turn_tensor,               # [1]
        "castling": castling_rights,       # [4]
        "ep": ep_tensor,                   # [16]
        "fiftymove": fiftymove_tensor,     # [1]
        "nbmove": nbmove_tensor,           # [1]
    }


def fen_to_feature_vector(fen: str) -> torch.Tensor:
    """
    Convert FEN string to a flat feature vector for neural network input.
    
    This is the main function used to prepare input for the model.
    
    Args:
        fen: Complete FEN string
        
    Returns:
        Tensor of shape [855] containing all position features:
            - 832 for board (64×13 flattened)
            - 1 for turn
            - 4 for castling rights
            - 16 for en passant
            - 1 for halfmove clock
            - 1 for fullmove number
            
    Example:
        >>> fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        >>> vec = fen_to_feature_vector(fen)
        >>> vec.shape
        torch.Size([855])
    """
    parsed = parse_fen(fen)
    flat_board = parsed["board"].view(-1)  # Flatten from [64, 13] to [832]
    
    return torch.cat(
        [
            flat_board,              # [832] Board position
            parsed["turn"],          # [1] Turn indicator
            parsed["castling"],      # [4] Castling rights
            parsed["ep"],            # [16] En passant square
            parsed["fiftymove"],     # [1] Halfmove clock
            parsed["nbmove"],        # [1] Fullmove number
        ]
    )


if __name__ == "__main__":
    # Test the encoding with sample data
    import json
    
    print("Testing FEN encoding...")
    print("=" * 60)
    
    # Test with starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"\nTest 1: Starting position")
    print(f"FEN: {starting_fen}")
    
    vec = fen_to_feature_vector(starting_fen)
    print(f"Feature vector shape: {vec.shape}")  # Should be [855]
    print(f"Feature vector dtype: {vec.dtype}")  # Should be float32
    print(f"Non-zero elements: {(vec != 0).sum().item()}")
    
    # Test with actual data file if available
    try:
        print(f"\nTest 2: Loading from data file")
        with open("data/processed_data/lichess-2025-07.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        
        fen = data[0]["FEN"]
        print(f"FEN: {fen}")
        
        vec = fen_to_feature_vector(fen)
        print(f"Feature vector shape: {vec.shape}")
        
        # Test parsing components
        parsed = parse_fen(fen)
        print(f"\nParsed components:")
        print(f"  Board shape: {parsed['board'].shape}")
        print(f"  Turn: {parsed['turn'].item()}")
        print(f"  Castling rights: {parsed['castling'].tolist()}")
        print(f"  En passant: {parsed['ep'].sum().item()} (1 if en passant possible)")
        print(f"  Halfmove clock: {parsed['fiftymove'].item()}")
        print(f"  Fullmove number: {parsed['nbmove'].item()}")
        
    except FileNotFoundError:
        print("Data file not found. Skipping file test.")
    
    print("\n" + "=" * 60)
    print("✓ FEN encoding test completed successfully!")