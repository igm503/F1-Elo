import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def get_season_urls(start_year, end_year):
    season_url_list = []
    for year in range(start_year, end_year + 1):
        season_url = f'https://www.racing-reference.info/season-stats/{year}/F/'
        season_url_list.append(season_url)
    return season_url_list

def get_race_urls(season_url):
    race_urls = []
    req = requests.get(season_url)
    content = BeautifulSoup(req.text, 'html.parser')
    row = content.find_all('div', {"class": "race-number"})
    for link in row[1:]:
        if link is not None:
            race_urls.append(link.find('a').get('href'))
    return race_urls

def get_urls(start_year, end_year):
    url_list = []
    print(f'Downloading Data from {start_year} to {end_year}')
    for season in tqdm(get_season_urls(start_year, end_year)):
        url_list.append(get_race_urls(season))
    return url_list

print(get_urls(1992, 2021))

