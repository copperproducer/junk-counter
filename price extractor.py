from selenium import webdriver
from bs4 import BeautifulSoup
import csv
import time

# Set up Selenium WebDriver (Chrome)
driver = webdriver.Chrome()

# New Warframe Wiki URL
url = "https://wiki.warframe.com/w/Ducats/Prices/All"

# Open the page with Selenium
driver.get(url)

# Wait for the page to load
time.sleep(3)

# Get the page source after JavaScript has loaded the content
page_source = driver.page_source

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(page_source, 'html.parser')

# Close the browser once the page has been loaded
driver.quit()

# Find the correct table using class name
table = soup.find('table', class_='listtable sortable jquery-tablesorter')

if table is None:
    print("Failed to find the target table.")
    exit()  # Exit if the table isn't found

# Extract data from the table
data = []
rows = table.find_all('tr')[1:]  # Skip the header row

for row in rows:
    cols = row.find_all('td')
    if len(cols) > 1:  # Ensure it's a valid data row
        part_name = cols[0].text.strip()  # First column: Part name
        ducat_value = cols[2].find('b').text.strip()  # Third column: Ducat value inside <b>

        data.append([part_name, ducat_value])

# Write data to CSV
with open('ducat_values.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Part Name', 'Value'])  # Headers
    writer.writerows(data)

print('Data has been written to ducat_values.csv')
