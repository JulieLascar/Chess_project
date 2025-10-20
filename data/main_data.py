import json
from pathlib import Path
from scripts.downloader import *
from scripts.parser import *
from scripts.tokenizer import *
from scripts.make_fen import *

# --- Charger la config ---
config_path = Path(__file__).parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

def main_data(config=config):
    # --- download data ---
    chessdataloader = ChessDataDownloader(url=config["download_url"])
    filepath = chessdataloader.download()
    png_path = chessdataloader.decompress(filepath)
    
    filepath = Path(filepath)
    if filepath.exists() and delete_data:
        filepath.unlink() 
        print(f"Fichier supprimé : {filepath}")

    # --- Filter png ---
    token_config = TokenizerConfig(
        pad_token='<PAD>',
        special_tokens=['<START>', '<END>'],
        save_path='./models/chess_tokenizer.json'
    )
    tokenizer = ChessTokenizer(token_config)
    png_parser= PGNParser(tokenizer= tokenizer,min_elo=config['min_elo'], min_moves = config['min_moves'], max_moves = config['max_moves'])
    games, valid_games, total_games = png_parser.parse_file(filepath=png_path, max_games = config['max_games'])
    data = {'metadata' : config, 'PGNs':games }

    os.makedirs('data/processed_data', exist_ok=True)
    pgnFiltered_path = Path(f"data/processed_data/pgnFiltered_{config['ref_name']}.json")
    with open(pgnFiltered_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print('Fichier pgnFiltered créé')
    
    png_path = Path(png_path)
    if png_path.exists() and delete_data:
        png_path.unlink() 
        print(f"Fichier supprimé : {png_path}")

    # --- Make fen data ---
    with open(pgnFiltered_path, "r", encoding="utf-8") as f:
        games = json.load(f)

    games = games['PGNs']

    with open(os.path.join('data/processed_data',f"{config['ref_name']}.jsonl"), "w", encoding="utf-8") as f:
        for game, pgn in enumerate(games):
            fens, moves, legal_moves = pgn2fen(pgn)
            for coup in range(len(fens)):
                line = {
                    'data_ref': config['ref_name'],
                    'id': f'{game}_{coup}',
                    'FEN': fens[coup],
                    'legal_moves': legal_moves[coup],
                    'human': {'next_move': moves[coup],}
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print('Fichier fen créé')

if __name__ == "__main__":
    delete_data = True # effacer les fichiers intermédiaires (lourds)
    main_data(config=config)
