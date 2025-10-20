# Chess Move Prediction with Neural Networks

A machine learning project for predicting chess moves using neural networks, with comprehensive Stockfish evaluation for performance analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project trains neural networks to predict chess moves from FEN (Forsyth-Edwards Notation) positions. It explores various training configurations, including:

- Different training dataset sizes
- Rule enforcement strategies (legal moves filtering)
- Comparison with Stockfish engine evaluations

The project uses games from the Lichess database and implements a complete pipeline from data download to model evaluation.

## ✨ Features

- **Automated Data Pipeline**: Download and process chess games from Lichess
- **Flexible Training**: Experiment with different dataset sizes and rule configurations
- **Rule Enforcement**: Train models with or without legal move constraints
- **Stockfish Integration**: Comprehensive position evaluation using Stockfish engine
- **Parallel Processing**: Multi-core support for faster Stockfish analysis
- **TensorBoard Support**: Real-time training visualization
- **Automatic Resume**: Continue interrupted processing from last checkpoint
- **Move Quality Analysis**: Evaluate prediction quality with centipawn loss metrics

## 📁 Project Structure

```
projet/
├── src/
│   ├── config.py           # Configuration management
│   ├── dataset.py          # PyTorch dataset for chess positions
│   ├── make_input.py       # FEN encoding to feature vectors
│   ├── model.py            # Neural network architectures
│   ├── train.py            # Training script
│   ├── inference.py        # Model inference
│   └── computeSF.py        # Stockfish evaluation engine
├── configs/
│   ├── train_config.json       # Training configuration
│   ├── inference_config.json   # Inference configuration
│   └── stockfish_config.json   # Stockfish analysis configuration
├── data/
│   ├── config.json         # Data download configuration
│   ├── main_data.py        # Data processing pipeline
│   └── scripts/            # Helper scripts for data processing
├── experiments/            # Experiment outputs and models
└── README.md
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Stockfish chess engine
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd projet
```

2. **Install Python dependencies**
```bash
pip install torch torchvision
pip install python-chess
pip install tqdm
```

3. **Install Stockfish**
```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# macOS
brew install stockfish

# Or download from https://stockfishchess.org/
```

4. **Configure Stockfish path**

Edit `STOCKFISH_PATH` in `src/computeSF.py`:
```python
STOCKFISH_PATH = "/usr/games/stockfish"
```

## 🚀 Usage

### 1. Data Preparation

Configure data parameters in `data/config.json`:
```json
{
  "download_url": "https://database.lichess.org/standard/...",
  "ref_name": "lichess-2025-07",
  "max_games": 1000,
  "min_elo": 2000,
  "min_moves": 20,
  "max_moves": 300
}
```

Download and process chess games from Lichess:

```bash
cd data
python main_data.py
```

This will:
- Download PGN games from Lichess
- Filter games based on ELO, move count, etc.
- Convert games to FEN positions
- Save processed data to `data/processed_data/`


### 2. Model Training

Train models with different configurations:

```bash
cd src
python train.py
```

Configure training in `configs/train_config.json`:
```json
{
  "train_data_path": "data/processed_data/lichess-2025-07.jsonl",
  "L_games_nb": [100, 1000, 10000],
  "model_name": "OLP",
  "lr": 0.001,
  "epochs": 50,
  "L_rules": ["norules", "withrules"],
  "tensorboard": true,
  "save_model": true
}
```

**Rule modes:**
- `norules`: Model predicts any move (may be illegal)
- `withrules`: Legal moves mask applied during training and evaluation
- `medium`: Legal moves mask applied only during evaluation

### 3. Model Inference

Generate predictions :

```bash
python inference.py
```

### 4. Stockfish Evaluation

Evaluate predictions with Stockfish:

```bash
python computeSF.py
```

This analyzes:
- Stockfish's best move for each position
- Human player moves
- Model predictions

## 🧠 Model Architecture

### One Layer Perceptron (OLP)

A simple baseline model consisting of:
- **Input**: 855-dimensional feature vector
  - 832 features: Board position (64 squares × 13 channels)
  - 1 feature: Turn (white/black)
  - 4 features: Castling rights
  - 16 features: En passant square
  - 2 features: Move counters
- **Output**: ~20,480 classes (all possible moves)

Despite its simplicity, OLP provides a strong baseline for move prediction tasks.

### Feature Encoding

**Board representation (64 × 13):**
- 12 channels for pieces (P, R, N, B, Q, K for white/black)
- 1 channel for empty squares

**Additional features:**
- Turn indicator
- Castling availability (KQkq)
- En passant target square
- Halfmove clock (fifty-move rule)
- Fullmove number

## 📊 Evaluation Metrics

### 

- **SF_score**: 
- **delta**: 


### Output Format

Results are saved in JSONL format with the following structure:

```json
{
  "data_ref": "lichess-2025-07",
  "id": "0_15",
  "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "legal_moves": ["e2e4", "d2d4", ...],
  "human": {
    "next_move": "e2e4",
    "SF_score": 25,
    "delta": 0
  },
  "stockfish": {
    "best_move": "e2e4",
    "SF_score": 25
  },
  "OLP": {
    "100_norules": {
      "prediction": "e2e4",
      "SF_score": 25,
      "delta": 0
    },
    "100_withrules": {
      "prediction": "d2d4",
      "SF_score": 20,
      "delta": -5
    }
  }
}
```

## 📈 Results

Training results are saved in:
- `experiments/<expe_name>/models/` - Trained model checkpoints
- `experiments/<expe_name>/runs/` - TensorBoard logs
- `experiments/<expe_name>/<ref_name>_inference.jsonl` - Model predictions
- `experiments/<expe_name>/<ref_name>_SF.jsonl` - Stockfish evaluations

### Viewing Training Progress

```bash
tensorboard --logdir experiments/<expe_name>/runs/
```

## 🔄 Workflow Summary

```
1. Download Data (main_data.py)
   ↓
2. Train Models (train.py)
   ↓
3. Generate Predictions (inference.py)
   ↓
4. Evaluate with Stockfish (computeSF.py)
   ↓
5. Analyze Results
```

## 🛠️ Advanced Features

### Parallel Processing

Stockfish analysis supports multiprocessing for faster evaluation:
```python
workers = 14 
```

### Caching

Stockfish evaluations are cached during analysis to avoid redundant computations for identical positions.

## 📝 Notes

- Stockfish depth is set to 15 by default (configurable in `computeSF.py`)
- Training uses Adam optimizer with default learning rate 0.001
- Models are saved when validation accuracy improves
- All configurations use JSON files for easy experimentation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional model architectures (CNN, Transformer)
- Opening book integration
- Endgame tablebases
- Advanced feature engineering
- Hyperparameter optimization
- Web interface for live prediction


##  Acknowledgments

- [Lichess](https://lichess.org/) for providing open chess game databases
- [Stockfish](https://stockfishchess.org/) chess engine
- [python-chess](https://python-chess.readthedocs.io/) library

---

**Happy chess predicting! ♟️**