import pandas as pd
import pytest
import numpy as np
from utils.transform import transform_to_DataFrame, transform_data

@pytest.fixture
def sample_data():
    return [
        {
            'Title': 'Product A',
            'Price': '$10.00',
            'Rating': 'Rating: 4.8 ⭐ / 5',
            'Colors': '2 colors',
            'Size': 'Size: L',
            'Gender': 'Gender: Male',
            'timestamp': '2024-01-01T00:00:00'
        },
        {
            'Title': 'Unknown Product',
            'Price': '$0.00',
            'Rating': 'Rating: 0 ⭐ / 5',
            'Colors': '0 colors',
            'Size': 'Size: XL',
            'Gender': 'Gender: Female',
            'timestamp': '2024-01-01T00:00:00'
        },
        {
            'Title': 'Product B',
            'Price': 'Price Unavailable',
            'Rating': 'Invalid Rating',
            'Colors': 'No colors',
            'Size': 'Invalid Size',
            'Gender': 'Unspecified',
            'timestamp': 'invalid date'
        }
    ]

@pytest.fixture
def transformed_data(sample_data):
    df = transform_to_DataFrame(sample_data)
    return transform_data(df, exchange_rate=15000)

def test_transform_to_DataFrame(sample_data):
    """Test DataFrame conversion with correct structure"""
    df = transform_to_DataFrame(sample_data)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {'Title', 'Price', 'Rating', 'Colors', 'Size', 'Gender', 'timestamp'}

def test_transform_to_DataFrame_empty_data():
    """Test handling of empty input data"""
    with pytest.raises(ValueError):
        transform_to_DataFrame([])

def test_transform_to_DataFrame_invalid_data():
    """Test handling of invalid input data"""
    with pytest.raises(ValueError):
        transform_to_DataFrame("not a list or dict")

def test_transform_data_removes_unknown_products(transformed_data):
    assert 'Unknown Product' not in transformed_data['Title'].values

def test_transform_data_missing_columns():
    """Test handling of DataFrames with missing columns"""
    test_data = [{'Title': 'P1', 'Price': '$100'}]
    df = transform_to_DataFrame(test_data)
    with pytest.raises(ValueError):
        transform_data(df, exchange_rate=1)

def test_transform_data_invalid_exchange_rate(sample_data):
    """Test handling of invalid exchange rates"""
    df = transform_to_DataFrame(sample_data)
    with pytest.raises(ValueError):
        transform_data(df, exchange_rate=0)  
    with pytest.raises(ValueError):
        transform_data(df, exchange_rate=-1)  
    with pytest.raises(ValueError):
        transform_data(df, exchange_rate="invalid")  

def test_transform_data_empty_dataframe():
    """Test handling of empty DataFrame after filtering"""
    test_data = [{'Title': 'Unknown Product', 'Price': '$10', 'Rating': 'Rating: 5 ⭐ / 5'}]
    df = transform_to_DataFrame(test_data)
    result = transform_data(df, exchange_rate=1)
    assert len(result) == 0  

def test_transform_data_raises_on_non_dataframe():
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        transform_data(["not", "a", "DataFrame"], exchange_rate=1.0)

def test_transform_data_raises_on_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        transform_data(empty_df, exchange_rate=1.0)

def test_price_parsing_exception():
    df = pd.DataFrame([{
        'Title': 'Test Shirt',
        'Price': '$invalid',
        'Rating': 'Rating: ⭐ 4.0 / 5',
        'Colors': '3 Colors',
        'Size': 'Size: M',
        'Gender': 'Gender: Women'
    }])
    result = transform_data(df, exchange_rate=1.0)
    assert result.empty

def test_rating_parsing_exception():
    df = pd.DataFrame([{
        'Title': 'Test Shirt',
        'Price': '$100',
        'Rating': 'Rating: ⭐ Invalid / 5',
        'Colors': '3 Colors',
        'Size': 'Size: M',
        'Gender': 'Gender: Women'
    }])
    result = transform_data(df, exchange_rate=1.0)
    assert result.empty

def test_colors_parsing_exception():
    df = pd.DataFrame([{
        'Title': 'Test Shirt',
        'Price': '$100',
        'Rating': 'Rating: ⭐ 4.5 / 5',
        'Colors': None,  # triggers isinstance check fail
        'Size': 'Size: M',
        'Gender': 'Gender: Women'
    }])
    result = transform_data(df, exchange_rate=1.0)
    assert result.iloc[0]['Colors'] == 0

def test_size_parsing_exception():
    df = pd.DataFrame([{
        'Title': 'Test Shirt',
        'Price': '$100',
        'Rating': 'Rating: ⭐ 4.5 / 5',
        'Colors': '3 Colors',
        'Size': 12345,
        'Gender': 'Gender: Women'
    }])
    result = transform_data(df, exchange_rate=1.0)
    assert result.iloc[0]['Size'] is None
