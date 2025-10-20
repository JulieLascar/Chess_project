"""
Stockfish analysis module.
Computes Stockfish evaluations for chess positions and model predictions.
Supports parallel processing and automatic resume from interruptions.
"""

import json
import os
import chess
import chess.engine
import multiprocessing
from tqdm import tqdm

# Stockfish configuration
STOCKFISH_PATH = "/usr/games/stockfish"
DEPTH = 15


def extract_score(info: dict, turn_white: bool = True):
    """
    Extract readable score from Stockfish analysis.
    
    Args:
        info: Stockfish analysis info dictionary
        turn_white: Whether it's white's turn to move
        
    Returns:
        Score in centipawns (int) or mate notation (str, e.g., "mate+3")
    """
    score_obj = info["score"].white() if turn_white else info["score"].black()
    
    if score_obj.is_mate():
        mate_in = score_obj.mate()
        return f"mate+{mate_in}" if mate_in > 0 else f"mate{mate_in}"
    
    return score_obj.score()


def analyse_entry(entry: tuple) -> dict:
    """
    Perform complete Stockfish analysis for a single position.
    
    Analyzes:
    - Stockfish's best move and evaluation
    - Human player's move and evaluation
    - Model predictions and evaluations
    
    Args:
        entry: Tuple containing (fen, legal_moves, human_data, olp_data, 
               stockfish_data, cache)
        
    Returns:
        Dictionary with analysis results for all moves
    """
    fen, lm, human_data, olp, stockfish_data, cache = entry

    board = chess.Board(fen)
    turn_white = board.turn == chess.WHITE
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    result = {
        "human": human_data.copy(),
        "stockfish": stockfish_data.copy(),
        "OLP": olp.copy(),
    }

    # Cache helper functions
    def get_cached_score(fen_str: str, move_str: str):
        """Retrieve score from cache if available."""
        return cache.get((fen_str, move_str))

    def set_cached_score(fen_str: str, move_str: str, score_value):
        """Store score in cache."""
        cache[(fen_str, move_str)] = score_value

    # --- Analyze Stockfish's best move ---
    if "best_move" not in stockfish_data or "SF_score" not in stockfish_data:
        cached = get_cached_score(fen, "best")
        
        if cached is not None:
            best_move, score_best = cached
        else:
            # Calculate best move
            engine.configure({"Clear Hash": True})
            info_best = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            best_move = info_best["pv"][0].uci()

            # Evaluate position after best move
            board_best = board.copy()
            board_best.push(chess.Move.from_uci(best_move))
            engine.configure({"Clear Hash": True})
            info_after_best = engine.analyse(
                board_best, chess.engine.Limit(depth=DEPTH)
            )
            score_best = extract_score(info_after_best, turn_white)

            set_cached_score(fen, "best", (best_move, score_best))

        result["stockfish"] = {"best_move": best_move, "SF_score": score_best}
    else:
        best_move = stockfish_data["best_move"]
        score_best = stockfish_data["SF_score"]

    # --- Analyze human move ---
    move = human_data.get("next_move")
    if move and ("SF_score" not in human_data or "delta" not in human_data):
        if move not in lm:
            # Illegal move
            score_human = "illegal"
            delta_human = None
        else:
            cached = get_cached_score(fen, move)
            
            if cached is not None:
                score_human = cached
            else:
                # Evaluate human move
                board_human = board.copy()
                board_human.push(chess.Move.from_uci(move))
                engine.configure({"Clear Hash": True})
                info_human = engine.analyse(
                    board_human, chess.engine.Limit(depth=DEPTH)
                )
                score_human = extract_score(info_human, turn_white)
                set_cached_score(fen, move, score_human)

            # Calculate delta (difference from best move)
            if not isinstance(score_human, str) and not isinstance(score_best, str):
                delta_human = score_human - score_best
            else:
                delta_human = None

        result["human"]["SF_score"] = score_human
        result["human"]["delta"] = delta_human

    # --- Analyze model predictions (OLP) ---
    olp_with_scores = {}
    
    for model_name, model_data in olp.items():
        # Normalize model data format
        if isinstance(model_data, str):
            pred_move = model_data
            model_data = {"prediction": pred_move}
        else:
            pred_move = model_data.get("prediction") or model_data.get("next_move")

        # Skip if already analyzed
        if "SF_score" in model_data and "delta" in model_data:
            olp_with_scores[model_name] = model_data
            continue

        if pred_move not in lm:
            # Illegal prediction
            score_pred = "illegal"
            delta_pred = None
        else:
            cached = get_cached_score(fen, pred_move)
            
            if cached is not None:
                score_pred = cached
            else:
                # Evaluate predicted move
                board_pred = board.copy()
                board_pred.push(chess.Move.from_uci(pred_move))
                engine.configure({"Clear Hash": True})
                info_pred = engine.analyse(
                    board_pred, chess.engine.Limit(depth=DEPTH)
                )
                score_pred = extract_score(info_pred, turn_white)
                set_cached_score(fen, pred_move, score_pred)

            # Calculate delta
            if not isinstance(score_pred, str) and not isinstance(score_best, str):
                delta_pred = score_pred - score_best
            else:
                delta_pred = None

        olp_with_scores[model_name] = {
            "prediction": pred_move,
            "SF_score": score_pred,
            "delta": delta_pred,
        }

    result["OLP"] = olp_with_scores
    engine.quit()
    return result


def process_file(
    input_path: str,
    output_path: str,
    workers: int,
    batch_size: int
):
    """
    Process JSONL file with Stockfish analysis.
    
    Features:
    - Automatic resume from interruptions
    - Parallel processing with multiprocessing
    - Progress tracking with tqdm
    - Shared cache for position evaluations
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        workers: Number of parallel worker processes
        batch_size: Number of positions per batch
    """
    # Check for existing progress
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as fout:
            already_done = sum(1 for _ in fout)
    print(f"⚡ Automatic resume: {already_done} lines already processed")

    # Create shared cache for multiprocessing
    cache = multiprocessing.Manager().dict()

    with open(input_path, "r") as fin:
        # Skip already processed lines
        for _ in range(already_done):
            fin.readline()

        batch = []
        line_buffer = []
        batch_num = already_done // batch_size

        for line in tqdm(fin, desc="Reading dataset"):
            obj = json.loads(line)
            fen = obj.get("FEN")
            lm = obj.get("legal_moves", [])
            human_data = obj.get("human", {})
            olp = obj.get("OLP", {})
            stockfish_data = obj.get("stockfish", {})

            batch.append((fen, lm, human_data, olp, stockfish_data, cache))
            line_buffer.append(obj)

            # Process batch when full
            if len(batch) >= batch_size:
                batch_num += 1
                print(f"\n--- Batch {batch_num} ({len(batch)} lines) ---")

                # Parallel processing
                with multiprocessing.Pool(processes=workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap(analyse_entry, batch),
                            total=len(batch),
                            desc="Stockfish analysis"
                        )
                    )

                # Write results
                with open(output_path, "a") as fout:
                    for obj, result in zip(line_buffer, results):
                        obj.update(result)
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                batch.clear()
                line_buffer.clear()

        # Process final batch
        if batch:
            batch_num += 1
            print(f"\n--- Final batch {batch_num} ({len(batch)} lines) ---")
            
            with multiprocessing.Pool(processes=workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(analyse_entry, batch),
                        total=len(batch),
                        desc="Stockfish analysis (final)"
                    )
                )

            with open(output_path, "a") as fout:
                for obj, result in zip(line_buffer, results):
                    obj.update(result)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from config import Config

    # Load configuration
    cfg = Config.from_json("configs/stockfish_config.json")

    # Setup paths
    input_path = f"experiments/{cfg.expe}/{cfg.ref_name}_inference.jsonl"
    output_path = f"experiments/{cfg.expe}/{cfg.ref_name}_SF.jsonl"
    
    # Run analysis
    process_file(
        input_path=input_path,
        output_path=output_path,
        workers=cfg.workers,
        batch_size=cfg.batch_size
    )
    
    print("\n✓ Stockfish analysis completed!")