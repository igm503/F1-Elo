import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pymc as pm

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
            car = results[0][i,4]
            elo_table[driver] = [[1000, 100]] #Mean and Std; Assuming Elo is Normal
            if car not in elo_table:
                elo_table[car] = [[1000, 100]]
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
                    elo_table[driver].append([prior_elos[driver][0] - time_loss, prior_elos[driver][1]])
                    print(driver, "was absent from race", race)
            elif driver in new_elos:
                elo_table[driver] = [[None, None] for i in range(race)]
                print(driver, "subbed in race", race)
                elo_table[driver].append(new_elos[driver])

    return elo_table

def race_elo(result, prior_elo_dict):
    new_elo = {}
    # Check size of race, make ordered list of drivers and cars,
    # and make dict of driver->car_index pairs
    driver_list = []
    car_list = []
    driver_car = {}
    for i in range(result.shape[0]):
        driver = result[i, 0]
        car = result[i, 4]
        if driver is None:
            num_drivers = i
            break
        num_drivers = result.shape[0]
        driver_list.append(driver)
        if car not in car_list:
            car_list.append(car)
        driver_car[driver] = car_list.index(car)
    print(driver_list, car_list, driver_car)
    # Generate Observed Data in Correct Format
    matchups = []
    for i in range(num_drivers):
            for j in range(i + 1, num_drivers):
                matchups.append([1])
    # Give new cars and drivers prior Elos
    for driver in driver_list:
        if driver not in prior_elo_dict:
            prior_elo_dict[driver] = [1000, 100]
    for car in car_list:
        if car not in prior_elo_dict:
            prior_elo_dict[car] = [1000, 100]

    # Calculate Posterior Elos
    with pm.Model():
        # Create list of unobserved Driver Elo RV's
        elo_list = []
        for i in range(num_drivers):
            driver = driver_list[i]
            prior_mu = prior_elo_dict[driver][0]
            prior_sigma = prior_elo_dict[driver][1]
            elo_list.append(pm.Normal(f'elo{driver}', mu=prior_mu, sigma=prior_sigma))

        car_elo_list = []
        for i in range(len(car_list)):
            car = car_list[i]
            prior_mu_car = prior_elo_dict[car][0]
            prior_mu_car = prior_elo_dict[car][1]
            car_elo_list.append(pm.Normal(f'elo{car}', mu=prior_mu_car, sigma=prior_mu_car))
        # Create lists of unobserved P RV's and observed game data RV's
        p_list = []
        data_list = []
        count = 0
        for i in range(num_drivers):
            for j in range(i + 1, num_drivers):
                elo_diff = elo_list[j] + car_elo_list[driver_car[driver_list[j]]] - elo_list[i] - car_elo_list[driver_car[driver_list[i]]]
                p_list.append(1 / (1 + 10**(elo_diff/400)))
                data_list.append(pm.Bernoulli(f'data_{i}{j}', p=p_list[-1], observed=matchups[count]))
                count += 1
        # Create Posterior Sample
        trace = pm.sample(return_inferencedata=False, draws=500, cores=1)
    # Populate new_elo dictionary with key driver name 
    # and value [posterior mean, posterior std]
    for driver in driver_list:
        sample = trace.get_values(f'elo{driver}')
        mean = sample.mean()
        std = sample.std()
        new_elo[driver] = [mean, std]
    # Populate new_elo dictionary with car data
    for car in car_list:
        sample = trace.get_values(f'elo{car}')
        mean = sample.mean()
        std = sample.std()
        new_elo[car] = [mean, std]
    print(new_elo)
    return new_elo


race_list = get_urls(2020, 2020)
results = get_results(race_list)
elo_table = update_elo(results)
final_sum = 0
print(elo_table)
for driver in elo_table:
    print(driver, elo_table[driver][0][0], elo_table[driver][-1][0])
    final_sum += elo_table[driver][-1][0]
print(final_sum / 20)

for driver in elo_table:
    elo_over_season = []
    for elo_item in elo_table[driver]:
        elo_over_season.append(elo_item[0])
    plt.plot(range(len(elo_over_season)), elo_over_season, label=driver)
plt.legend()
plt.show()


#Placket Luce (is it basically the same as pairwise ELO?)

#Model based on probability of passing or being passed by each other driver? 
