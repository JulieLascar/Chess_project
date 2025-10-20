"""
PGN to FEN conversion module.
Converts chess games in PGN format to sequences of FEN positions with moves.

This module processes PGN games move by move, generating:
- FEN notation for each position
- The move played from that position
- All legal moves available in that position
"""

import io
import chess.pgn


def pgn2fen(pgn: str) -> tuple[list, list, list]:
    """
    Convert a PGN game string to a sequence of FEN positions with moves.
    
    Processes each move in the game, generating the FEN position before the move,
    the move played, and all legal moves available in that position.
    
    Note: The final position (after the last move) is excluded since there is
    no next move to predict from that position.
    
    Args:
        pgn: Complete PGN game as a string
        
    Returns:
        Tuple containing three lists:
            - fens: List of FEN strings for each position (excluding final position)
            - moves: List of moves played from each position (in UCI notation)
            - legal_moves: List of lists, each containing legal moves for that position
        
    Example:
        >>> pgn = "1. e4 e5 2. Nf3 Nc6"
        >>> fens, moves, legal_moves = pgn2fen(pgn)
        >>> print(fens[0])  # Starting position FEN
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        >>> print(moves[0])  # First move in UCI notation
        'e2e4'
        >>> print(len(legal_moves[0]))  # Number of legal moves at start
        20
    """
    # Parse PGN string
    pgn_io = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_io)
    
    # Initialize board at starting position
    board = game.board()
    
    # Store starting position
    fens = [board.fen()]
    moves = []
    legal_moves = [[move.uci() for move in board.legal_moves]]

    # Iterate through all moves in the game
    for move in game.mainline_moves():
        # Store the move that was played (in UCI notation)
        moves.append(move.uci())
        
        # Apply the move to get next position
        board.push(move)
        
        # Store the new position and its legal moves
        fens.append(board.fen())
        legal_moves.append([move.uci() for move in board.legal_moves])
    
    # Exclude the final position (no next move to predict)
    return fens[:-1], moves, legal_moves[:-1]


if __name__ == "__main__":
    import json

    print("Testing PGN to FEN conversion...")
    print("=" * 60)
    
    # Reference name for the dataset
    ref = "lichess-2025-07"

    # Load filtered PGN games
    print(f"\nLoading games from: pgnFiltered_{ref}.json")
    try:
        with open(f"pgnFiltered_{ref}.json", "r", encoding="utf-8") as f:
            games_data = json.load(f)
    except FileNotFoundError:
        print("Error: PGN file not found. Using sample data for testing.")
        # Create sample data for testing
        sample_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
        print(f"\nTesting with sample game: {sample_pgn}")
        fens, moves, legal_moves = pgn2fen(sample_pgn)
        
        print(f"\nResults:")
        print(f"  Positions: {len(fens)}")
        print(f"  Moves: {len(moves)}")
        print(f"  Legal moves lists: {len(legal_moves)}")
        
        print(f"\nFirst position:")
        print(f"  FEN: {fens[0]}")
        print(f"  Move played: {moves[0]}")
        print(f"  Legal moves: {len(legal_moves[0])} moves")
        print(f"  Sample legal moves: {legal_moves[0][:5]}")
        
        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
        exit(0)

    metadata = games_data['metadata']
    games = games_data['PGNs']
    
    print(f"Found {len(games)} games")
    print(f"Metadata: {metadata}")

    # Convert games to annotated data (FEN + moves)
    output_file = f"annot_data_{ref}.jsonl"
    print(f"\nConverting games to JSONL format: {output_file}")
    
    total_positions = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for game_id, pgn in enumerate(games):
            # Convert PGN to FEN sequences
            fens, moves, legal_moves = pgn2fen(pgn)
            
            # Create one entry for each position in the game
            for move_id in range(len(fens)):
                line = {
                    'data_ref': ref,
                    'id': f'{game_id}_{move_id}',
                    'FEN': fens[move_id],
                    'next_move': moves[move_id],
                    'legal_moves': legal_moves[move_id],
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                total_positions += 1
            
            # Progress update every 100 games
            if (game_id + 1) % 100 == 0:
                print(f"  Processed {game_id + 1}/{len(games)} games...")
    
    print(f"\n✓ Conversion completed!")
    print(f"  Total games: {len(games)}")
    print(f"  Total positions: {total_positions}")
    print(f"  Average positions per game: {total_positions / len(games):.1f}")
    print(f"  Output saved to: {output_file}")
    print("=" * 60)