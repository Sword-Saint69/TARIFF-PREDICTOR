import pandas as pd
import joblib
import os

def load_data(base_path='/Users/gouthamsankar/Linux/maand'):
    """
    Load model, scaler and processed data
    
    Returns:
        tuple: (model, scaler, processed_data)
    """
    # Updated paths to look in the trained folder
    model = joblib.load(os.path.join(base_path, 'trained', 'best_trade_deficit_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'trained', 'scaler.pkl'))
    processed_data = pd.read_csv(os.path.join(base_path, 'processed_data.csv'))
    
    return model, scaler, processed_data