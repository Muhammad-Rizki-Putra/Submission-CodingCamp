import pandas as pd
import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock
from utils.load import store_to_postgre, store_to_googlesheet, store_to_csv

# Dummy DataFrame untuk pengujian
@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"Title": "Produk A", "Price": "10000", "Size": "M", "Rating": "4.5", "Gender": "Men"},
        {"Title": "Produk B", "Price": "20000", "Size": "L", "Rating": "4.0", "Gender": "Women"}
    ])


# ---------------- TEST: PostgreSQL ----------------
@patch("utils.load.create_engine")
def test_store_to_postgre_success(mock_create_engine, sample_df):
    mock_connection = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection
    mock_create_engine.return_value = mock_engine

    store_to_postgre(sample_df, "postgresql://user:pass@localhost/db")     

    mock_create_engine.assert_called_once()
    mock_connection.execute.assert_not_called() 

@patch("utils.load.create_engine")
def test_store_to_postgre_failure(mock_create_engine, sample_df):
    mock_create_engine.side_effect = Exception("Connection failed")
    
    with patch('builtins.print') as mock_print:
        store_to_postgre(sample_df, "invalid_connection_string")
        
        mock_print.assert_called_with("Terjadi kesalahan saat menyimpan ke PostgreSQL: Connection failed")

@patch("utils.load.create_engine")
def test_store_to_postgre_connection(mock_create_engine, sample_df):
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    mock_connection = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection
    
    store_to_postgre(sample_df, "postgresql://user:pass@localhost/db")
    
    mock_create_engine.assert_called_once_with("postgresql://user:pass@localhost/db")
    mock_engine.connect.assert_called_once()


# ---------------- TEST: Google Sheets ----------------
@patch("utils.load.build")
@patch("utils.load.Credentials.from_service_account_file")
def test_store_to_googlesheet_success(mock_creds, mock_build, sample_df):
    mock_service = MagicMock()
    mock_build.return_value = mock_service
    mock_sheet = mock_service.spreadsheets.return_value
    mock_sheet.values.return_value.update.return_value.execute.return_value = {}

    store_to_googlesheet(sample_df)

    mock_build.assert_called_once()
    mock_creds.assert_called_once()
    mock_sheet.values.return_value.update.assert_called_once()

@patch("utils.load.build")
@patch("utils.load.Credentials.from_service_account_file")
def test_store_to_googlesheet_failure(mock_creds, mock_build, sample_df):
    mock_build.side_effect = Exception("API Error")
    
    with patch('builtins.print') as mock_print:
        store_to_googlesheet(sample_df)
        
        mock_print.assert_called_with("Terjadi kesalahan saat menyimpan ke Google Sheet: API Error")


# ---------------- TEST: CSV ----------------
def test_store_to_csv_success(sample_df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        filename = tmp.name
    try:
        store_to_csv(sample_df, filename)
        assert os.path.exists(filename)
        df_loaded = pd.read_csv(filename)
        assert not df_loaded.empty
    finally:
        os.remove(filename)

def test_store_to_csv_content(sample_df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        filename = tmp.name
    
    try:
        store_to_csv(sample_df, filename)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        expected_header = "Title,Price,Size,Rating,Gender"
        assert expected_header in content
        assert "Produk A,10000,M,4.5,Men" in content
        assert "Produk B,20000,L,4.0,Women" in content
    finally:
        os.remove(filename)