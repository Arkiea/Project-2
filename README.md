# Project-2

## Installation
```bash
conda create -n project_2 python=3.9.7
conda activate project_2
pip install -r requirements.txt
```

## Demo

There are two ways to train the models:

### 1. Brute force various models using Python scripts

```bash
cd /path/to/Project-2
conda activate project_2
# Train the models and save the results
python src/run_xgb.py
# Find the best models and save as a csv
python src/find_best_model.py
```

### 2. Custom model with all indicators

Open [src/trading_bot.ipynb](src/trading_bot.ipynb)
