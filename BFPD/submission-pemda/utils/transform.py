import pandas as pd
import numpy as np
import re

def transform_to_DataFrame(data):
    """Convert data to DataFrame."""
    if not data:
        raise ValueError("Input data cannot be empty")
    return pd.DataFrame(data)

def transform_data(df, exchange_rate):
    """Transform the entire DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(exchange_rate, (int, float)) or exchange_rate <= 0:
        raise ValueError("Exchange rate must be a positive number")
    
    required_columns = ['Title', 'Price', 'Rating']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    optional_columns = ['timestamp', 'Colors', 'Size', 'Gender']
    for col in optional_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    if 'timestamp' in df.columns:
        if df['timestamp'].notna().any(): 
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df.reset_index(drop=True, inplace=True)

    def transform_row(row):
        price = str(row['Price'])
        if price == 'nan' or price == 'Price Unavailable':
            row['Price'] = np.nan
        else:
            try:
                row['Price'] = float(price.replace('$', '').replace(',', '')) * exchange_rate
            except:
                row['Price'] = np.nan
        
        rating = str(row['Rating'])
        if rating == 'nan':
            row['Rating'] = np.nan
        else:
            try:
                match = re.search(r'⭐\s*([\d.]+)', rating)
                if match and match.group(1).replace('.', '', 1).isdigit():
                    row['Rating'] = float(match.group(1))
                else:
                    row['Rating'] = np.nan
            except:
                row['Rating'] = np.nan
        
        # color 
        colors = row['Colors']
        if isinstance(colors, str):
            try:
                num = colors.split()[0]
                row['Colors'] = int(num) if num.isdigit() else 0
            except:
                row['Colors'] = 0
        else:
            row['Colors'] = 0
            
        # size
        size = row['Size']
        if isinstance(size, str):
            try:
                row['Size'] = size.replace('Size: ', '').strip()
            except:
                row['Size'] = None
        else:
            row['Size'] = None
            
        # gender
        gender = row['Gender']
        if isinstance(gender, str):
            try:
                row['Gender'] = gender.replace('Gender: ', '').strip()
            except:
                row['Gender'] = None
        else:
            row['Gender'] = None
            
        return row
    

    df = df.apply(transform_row, axis=1)
    df = df[df['Price'].notna() & df['Rating'].notna()]
    
    return df