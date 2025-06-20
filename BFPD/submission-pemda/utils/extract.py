import requests
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    )
}

def extract_product_data(section):
    """Ekstrak data produk dari elemen HTML."""
    try:
        details = section.find('div', class_='product-details')
        if not details:
            return None

        title = details.find('h3', class_='product-title').text.strip()
        price = details.find(['span', 'p'], class_='price').text.strip()

        rating = None
        colors = None
        size = None
        gender = None

        p_tags = details.find_all('p')
        for p in p_tags:
            text = p.text.strip()
            if text.startswith('Rating:'):
                rating = text
            elif 'Color' in text:
                colors = text
            elif text.startswith('Size:'):
                size = text
            elif text.startswith('Gender:'):
                gender = text

        timestamp = datetime.now().isoformat()

        return {
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Colors": colors,
            "Size": size,
            "Gender": gender,
            "timestamp": timestamp
        }

    except requests.exceptions.RequestException as e:
        print("Error fetching from website:", e)
        return None
    except Exception as e:
        print("Error extracting data:", e)
        return None



def fetch_page_content(url):
    """Mengambil konten HTML dari URL dengan user-agent yang ditentukan."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error saat mengambil {url}: {e}")
        return None

def scrape_multiple_pages(base_url, total_pages=50):
    """Menyaring data produk dari beberapa halaman (misal 50 halaman)."""
    all_data = []
    
    for page_number in range(1, total_pages + 1):
        if page_number == 1:
            url = base_url
        else:
            url = f"{base_url}page{page_number}"  
        print(f"Scraping page: {page_number}")  
        content = fetch_page_content(url)
        
        if not content:
            continue
        
        soup = BeautifulSoup(content, 'html.parser')
        products = soup.find_all('div', class_='collection-card')

        for product in products:
            product_data = extract_product_data(product)
            if product_data:
                all_data.append(product_data)
    
    return all_data
