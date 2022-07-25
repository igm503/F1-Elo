import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_season_urls(start_year, end_year):
    season_url_list = []
    for year in range(start_year, end_year + 1):
        season_url = f'https://www.racing-reference.info/season-stats/{year}/F/'
        season_url_list.append(season_url)
    return season_url_list

def get_race_urls(html):
    race_urls = []
    content = BeautifulSoup(html.text, 'html.parser')
    row = content.find_all('div', {"class": "race-number"})
    for link in row[1:]:
        if link is not None:
            race_urls.append(link.find('a').get('href'))
    return race_urls

def get_urls(start_year, end_year):
    url_list = []
    print(f'Downloading Data from {start_year} to {end_year}')
    session = requests.Session()
    for season in tqdm(get_season_urls(start_year, end_year)):
        url_list += get_race_urls(session.get(season))
    return url_list

def get_race(html):
    """Returns 20 x 7 array containing Driver's name, result, start position,
    team, car, laps completed, and reason for DNF if DNF or time if not DNF. Input 
    url points to the racing-reference url that contains the tabular data for the race.
    """
    content = BeautifulSoup(html.text, 'html.parser')
    table = content.find('table', {"class": "tb race-results-tbl"})
    rows = table.find_all('tr')
    #Data Structure: Driver / Result / Starting Position / Team / Car / Laps Completed / Reason for DNF
    results = np.empty((len(rows)-1, 7), dtype=object)
    #Dict for mapping HTML info to Data Structure
    index = {1:1, 2:2, 4:3, 5:4, 6:5, 7: 6}
    #Generate results data
    row_counter = 0
    for row in rows:
        if row['class'] != ['newhead']:
            col_counter = 1
            for col in row.find_all('td'):
                #exception for driver names
                if col.string is None:
                    results[row_counter, 0] = list(col.descendants)[-2]
                else:
                    if col_counter in index:
                        results[row_counter, index[col_counter]] = col.string
                    col_counter += 1
                if col_counter > 8:
                    break
            row_counter += 1
    return results

def get_results(url_list):
    """Creates 3d array of race results from the urls in the list input. Fetches
    results using the get_race function"""

    results_list = []
    print("Downloading Race Data")
    session = requests.Session()
    for url in tqdm(url_list):
        results_list.append(get_race(session.get(url)))
    return results_list
    
def update_elo(results):
    time_loss = 4
    elo_table = {}
    for i in range(results[0].shape[0]):
        if results[0][i, 0] is not None:
            driver = results[0][i,0]
            elo_table[driver] = [1000]
    prior_elos = {}
    for race in tqdm(range(len(results))):
        for driver in elo_table:
            prior_elos[driver] = elo_table[driver][-1]
        new_elos = race_elo(results[race], prior_elos)
        for driver in elo_table | new_elos :
            if driver in elo_table:
                if driver in new_elos:
                    elo_table[driver].append(new_elos[driver])
                else:
                    elo_table[driver].append(prior_elos[driver] - time_loss)
                    print(driver, "was absent from race", race)
            elif driver in new_elos:
                elo_table[driver] = [None for i in range(race)]
                print(driver, "subbed in race", race)
                elo_table[driver].append(new_elos[driver])

    return elo_table

def race_elo(result, prior_elo_dict):
    new_elo = {}
    #check size of race 
    for i in range(result.shape[0]):
        if result[i,0] is None:
            stop = i
            break
        else:
            stop = result.shape[0]
    #Take care of new drivers
    for i in range(stop):
        driver = result[i,0]
        if driver not in prior_elo_dict:
            prior_elo_dict[driver] = 1000
    for i in range(stop):
        driver = result[i,0]
        prior_elo = prior_elo_dict[driver]

        win_table = np.zeros((stop-1,2))
        for t in range(stop-1):
            win_table[t,0] = int(i <= t)
            if t < i:
                competitor_name = result[t, 0]
            else:
                competitor_name = result[t + 1, 0]
            win_table[t, 1] = prior_elo_dict[competitor_name]
        new_elo[driver] = posterior_elo(prior_elo, win_table)
    return new_elo

def posterior_elo(prior_elo, win_table):
    k = 4
    elo2 = win_table[:, 1]
    win = win_table[:, 0]
    p = 1 / (1 + 10**((elo2 - prior_elo)/400))
    posterior_elo = prior_elo + k * np.sum(win - p**win * (1-p)**(1-win))
    return posterior_elo


race_list = get_urls(1972, 2021)
results = get_results(race_list)
elo_table = update_elo(results)
final_sum = 0
for driver in elo_table:
    print(driver, elo_table[driver][0], elo_table[driver][-1])
    final_sum += elo_table[driver][-1]
print(final_sum / 20)

for driver in elo_table:
    plt.plot(range(len(elo_table[driver])), elo_table[driver], label='hi')

plt.show()


# ELO with k-factor updating?

#Placket Luce (is it basically the same as pairwise ELO?)

#Model based on probability of passing or being passed by each other driver? 
