from sqlalchemy import create_engine
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
 
def store_to_postgre(data, db_url):
    """Fungsi untuk menyimpan data ke dalam PostgreSQL."""
    try:
        engine = create_engine(db_url)
        
        with engine.connect() as con:
            data.to_sql('products', con=con, if_exists='append', index=False)
            print("Data berhasil ditambahkan ke PostgreSQL")
    
    except Exception as e:
        print(f"Terjadi kesalahan saat menyimpan ke PostgreSQL: {e}")

def store_to_googlesheet(data):
    """Fungsi untuk menyimpan data ke dalam Google Sheet."""
    SERVICE_ACCOUNT_FILE = './high-office-459714-j7-06a32a1049d6.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    credential = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    SPREADSHEET_ID = '1pDczeCHScLC3Ye_cekSFP6Fc9ZuWJA1t9VxFhskefnE'
    RANGE_NAME = 'Sheet1!A1'

    try:
        service = build('sheets', 'v4', credentials=credential) 
        sheet = service.spreadsheets()

        values = [data.columns.tolist()] + data.astype(str).values.tolist()

        body = {
            'values': values
        }

        result = sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            valueInputOption='RAW',
            body=body
        ).execute()

        print("Data berhasil ditambahkan ke Google Sheet")
    except Exception as e:
         print(f"Terjadi kesalahan saat menyimpan ke Google Sheet: {e}")


def store_to_csv(data, filename):
    try:
        with open(filename, mode='w', newline='') as file:
            data.to_csv(file, index=False)
        print("Data berhasil ditambahkan ke CSV")
    except Exception as e:
        print(f"Terjadi kesalahan saat menyimpan ke CSV: {e}")

