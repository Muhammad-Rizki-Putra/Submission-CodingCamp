import unittest
from utils.extract import extract_product_data, fetch_page_content, scrape_multiple_pages
from bs4 import BeautifulSoup
from unittest.mock import patch, Mock
import requests

class TestExtract(unittest.TestCase):
    def test_extract_product_data_valid(self):
        html = '''
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">Test Product</h3>
                <div class="price-container"><span class="price">$123.45</span></div>
                <p>Rating: ⭐ 4.9 / 5</p>
                <p>2 Colors</p>
                <p>Size: XL</p>
                <p>Gender: Men</p>
            </div>
        </div>
        '''
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find('div', class_='collection-card')
        data = extract_product_data(section)
        self.assertEqual(data['Title'], 'Test Product')
        self.assertEqual(data['Price'], '$123.45')
        self.assertIn('Rating', data['Rating'])
        self.assertEqual(data['Size'], 'Size: XL')

    def test_extract_product_data_missing_fields(self):
        html = '''
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">Minimal Product</h3>
                <div class="price-container"><span class="price">$10.00</span></div>
            </div>
        </div>
        '''
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find('div', class_='collection-card')
        data = extract_product_data(section)
        self.assertEqual(data['Title'], 'Minimal Product')
        self.assertEqual(data['Price'], '$10.00')
        self.assertIsNone(data['Rating'])
        self.assertIsNone(data['Colors'])
        self.assertIsNone(data['Size'])
        self.assertIsNone(data['Gender'])

    def test_extract_product_data_error_handling(self):
        data = extract_product_data(None)
        self.assertIsNone(data)

    def test_extract_product_data_no_details_div(self):
        html = '''
        <div class="collection-card">
            <!-- Tidak ada div.product-details -->
            <div class="wrong-details">
                <h3 class="product-title">Broken Product</h3>
            </div>
        </div>
        '''
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find('div', class_='collection-card')
        data = extract_product_data(section)
        self.assertIsNone(data)
    
    def test_extract_product_data_with_colors(self):
        html = '''
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">Colorful Product</h3>
                <div class="price-container"><span class="price">$75.00</span></div>
                <p>Color: Red, Blue</p>
            </div>
        </div>
        '''
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find('div', class_='collection-card')
        data = extract_product_data(section)
        self.assertEqual(data['Colors'], 'Color: Red, Blue')
    
    def test_extract_product_data_raises_exception(self):
        class Dummy:
            def find(self, *args, **kwargs):
                raise Exception("Test exception")

        dummy_section = Dummy()
        data = extract_product_data(dummy_section)
        self.assertIsNone(data)


    @patch("utils.extract.requests.get")
    def test_fetch_page_content_success(self, mock_get):
        mock_response = Mock()
        mock_response.content = b"<html>test</html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_page_content("http://example.com")
        self.assertEqual(result, b"<html>test</html>")

    @patch("utils.extract.requests.get")
    def test_fetch_page_content_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Network Error")

        result = fetch_page_content("http://example.com")
        self.assertIsNone(result)

    @patch("utils.extract.fetch_page_content")
    @patch("utils.extract.extract_product_data")
    def test_scrape_multiple_pages(self, mock_extract, mock_fetch):
        html_content = '''
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">Mock Product</h3>
                <div class="price-container"><span class="price">$10.00</span></div>
                <p>Rating: ⭐ 4.5 / 5</p>
                <p>1 Color</p>
                <p>Size: M</p>
                <p>Gender: Men</p>
            </div>
        </div>
        '''
        mock_fetch.return_value = html_content
        mock_extract.return_value = {
            "Title": "Mock Product",
            "Price": "$10.00",
            "Rating": "⭐ 4.5 / 5",
            "Colors": "1 Color",
            "Size": "Size: M",
            "Gender": "Gender: Men"
        }

        from utils.extract import scrape_multiple_pages
        result = scrape_multiple_pages("http://example.com/", total_pages=2)

        self.assertEqual(len(result), 2) 

if __name__ == '__main__':
    unittest.main()
