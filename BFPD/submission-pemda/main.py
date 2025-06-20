from utils.extract import scrape_multiple_pages
from utils.transform import transform_to_DataFrame
from utils.transform import transform_data
from utils.load import store_to_postgre
from utils.load import store_to_googlesheet
from utils.load import store_to_csv

def main():
    url = 'https://fashion-studio.dicoding.dev/'
    db_url = 'postgresql+psycopg2://rizki:supersecretpassword@localhost:5432/mepsdb'
    clothing_data = scrape_multiple_pages(url, 50)
    clothing_df = transform_to_DataFrame(clothing_data)

    clothing_df = transform_data(clothing_df,16000)
    store_to_postgre(clothing_df,db_url)
    store_to_googlesheet(clothing_df)
    store_to_csv(clothing_df, 'products.csv')
    print(clothing_df.info())

if __name__ == "__main__":
    main()