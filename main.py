"""
AutoJudge: Main Entry Point
Run this script to retrain the models.
"""
from src.train import train_pipeline

if __name__ == "__main__":
    print("Initializing AutoJudge System...")
    train_pipeline()
    print("Done! You can now run 'streamlit run app.py' to launch the interface.")
