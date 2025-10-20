"""https://github.com/l-pommeret/chess_char"""

import os
import requests
import zstandard as zstd
from tqdm import tqdm

class ChessDataDownloader:
    """Télécharge et décompresse les fichiers de données d'échecs."""
    
    def __init__(self, url: str, save_dir: str = "."):
        self.url = url
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download(self) -> str:
        """Télécharge le fichier avec une barre de progression."""
        filename = os.path.basename(self.url)
        filepath = os.path.join(self.save_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Le fichier {filename} existe déjà.")
            return filepath
            
        print(f"Téléchargement de {filename}...")
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(8192):
                    size = f.write(chunk)
                    pbar.update(size)
                    
        return filepath

    def decompress(self, filepath: str) -> str:
        """Décompresse un fichier .zst."""
        output_path = filepath.rsplit('.', 1)[0]
        
        if os.path.exists(output_path):
            print(f"Le fichier décompressé {output_path} existe déjà.")
            return output_path
            
        print(f"Décompression de {filepath}...")
        with open(filepath, 'rb') as compressed:
            with open(output_path, 'wb') as decompressed:
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(compressed, decompressed)
                
        return output_path
    
if __name__ == '__main__' : 
    url = 'https://database.lichess.org/standard/lichess_db_standard_rated_2025-07.pgn.zst'
    chessdataloader = ChessDataDownloader(url=url)
    filepath = chessdataloader.download()
    output_path = chessdataloader.decompress('lichess_db_standard_rated_2025-07.pgn.zst')