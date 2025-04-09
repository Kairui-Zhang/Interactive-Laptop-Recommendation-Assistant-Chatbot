# Scrape more pages: Change range(1, 6) to a larger range, such as range(1, 11)
# Keyword expansion: In addition to pagination, scrape using multiple keywords, such as laptop, gaming laptop, ultrabook
# Extract more fields: Extract additional details from the product page (e.g., brand, screen size, battery life, etc.)

# Standard libraries
import sqlite3
import random
import time

# Third-party libraries
import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------

def categorize_laptop(gpu, ram, weight, price):
    tags = []
    if "RTX" in gpu or "GTX" in gpu:
        tags.append("Gaming")
    elif "Iris" in gpu or "Integrated" in gpu:
        tags.append("Office")
    else:
        tags.append("General")
    if ram >= 16:
        tags.append("High-End")
    elif ram >= 8:
        tags.append("Student")
    else:
        tags.append("Budget")
    if weight < 1.5:
        tags.append("Portable")
    else:
        tags.append("Gaming")
    if price > 1500:
        tags.append("High-End")
    elif 800 <= price <= 1500:
        tags.append("Mid-Range")
    else:
        tags.append("Budget")
    return ", ".join(tags)

conn = sqlite3.connect('laptop_recommendations_4.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS laptops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    price REAL,
    gpu TEXT,
    ram INTEGER,
    weight REAL,
    link TEXT,
    tags TEXT
)
''')

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
]
headers = {
    'User-Agent': random.choice(user_agents),
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.amazon.com/',
    'Connection': 'keep-alive'
}

base_url = 'https://www.amazon.com/s?k=laptop&page={}'
for page in range(1, 11):
    try:
        print(f"Fetching page {page}...")
        url = base_url.format(page)
        time.sleep(random.uniform(5, 10)) # Request interval
        response = requests.get(url, headers=headers)

        if "Type the characters you see below" in response.text:
            print("CAPTCHA page detected. Skipping...")
            continue

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            print(soup.prettify()[:1000]) # Check HTML content
            products = soup.find_all('div', {'data-component-type': 's-search-result'})

            for product in products:
                try:
                    model = product.h2.text.strip() if product.h2 else "Unknown Model"
                    price_tag = product.find('span', class_='a-price-whole')
                    price = float(price_tag.text.replace(",", "")) if price_tag else 0 # Default price is 0
                    link_tag = product.h2.a if product.h2 else None
                    link = 'https://www.amazon.com' + link_tag['href'] if link_tag else "Unknown Link"

                    gpu = "RTX 3050" if "Gaming" in model else "Integrated"
                    ram = 16 if "Gaming" in model else 8
                    weight = 2.5 if "Gaming" in model else 1.2

                    tags = categorize_laptop(gpu, ram, weight, price)
                    cursor.execute('''
                    SELECT COUNT(*) FROM laptops WHERE model = ? AND price = ?
                    ''', (model, price))
                    if cursor.fetchone()[0] == 0:
                        cursor.execute('''
                        INSERT INTO laptops (model, price, gpu, ram, weight, link, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (model, price, gpu, ram, weight, link, tags))
                except Exception as e:
                    print(f"Error parsing product: {e}")
        else:
            print(f"Failed to fetch page {page}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching page {page}: {e}")

conn.commit()
conn.close()
print("Data fetching complete.")
