import os
import warnings

# Global Configuration
DATA_PATH = 'data/problem_data.jsonl'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
LOG_FILE = os.path.join(REPORTS_DIR, 'experiment_log.txt')

def setup_directories():
    """Creates necessary folders and clears warnings."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def setup_logging():
    """Initializes the log file."""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("--- AutoJudge Experiment Log ---\n\n")

def log_message(message):
    """Prints to console and appends to log file."""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + "\n")