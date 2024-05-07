import requests
from bs4 import BeautifulSoup
import csv

# URL of the page to scrape
url = "https://warframe.fandom.com/wiki/Orokin_Ducats#Prices"

# Send an HTTP request to the URL
response = requests.get(url)
# Parse the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')

# Attempt to find the correct table by checking for a header containing specific text
target_table = None
tables = soup.find_all('table')
for table in tables:
    headers = [header.text.strip() for header in table.find_all('th')]
    if "Part" in headers and "Ducat Value" in headers:
        target_table = table
        break

if target_table is None:
    print("Failed to find the target table.")
    exit()  # Exit the script if the table isn't found

# Extract data from the identified table
data = []
# Find all rows in the table
rows = target_table.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    if len(cols) > 1:  # to ensure it's a data containing row
        part_name = cols[0].text.strip()
        ducat_value = cols[-1].text.strip()  # Assuming ducat value is in the last column
        data.append([part_name, ducat_value])

# Write data to a CSV file
with open('parts_and_values.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Part Name', 'Ducat Value'])  # Writing headers
    writer.writerows(data)

print('Data has been written to parts_and_values.csv')
