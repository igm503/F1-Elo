import pymc as pm
import matplotlib.pyplot as plt

games_12 = [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
games_34 = [1, 1]
games_23 = [1, 0, 0, 0, 0, 0, 0, 0]

def four_players(games_12, games_23, games_34):
    with pm.Model():
        elo1 = pm.Normal('elo1', mu=1500, sigma=100)
        elo2 = pm.Normal('elo2', mu=1500, sigma=100)
        elo3 = pm.Normal('elo3', mu=1500, sigma=100)
        elo4 = pm.Normal('elo4', mu=1500, sigma=100)
        p_12 = 1 / (1 + 10**((elo2 - elo1) / 400))
        p_34 = 1 / (1 + 10**((elo4 - elo3) / 400))
        p_23 = 1 / (1 + 10**((elo3 - elo2) / 400))

        data_12 = pm.Bernoulli('data_12', p=p_12, observed=games_12)
        data_34 = pm.Bernoulli('data_34', p=p_34, observed=games_34)
        data_23 = pm.Bernoulli('data_23', p=p_23, observed=games_23)
        trace = pm.sample(return_inferencedata=False, draws=5000, cores=1)
        print('context over\n\n')

    for i in range(4):
        sample = trace.get_values(f'elo{i+1}')
        mean = sample.mean()
        std = sample.std()
        print(f'Driver {i+1} Elo: Mean = {mean}, Std = {std}')


def n_players(n, games):
    num_players = n
    with pm.Model():
        print('beginnign of model')
        # Create list of unobserved Elo RV's
        elo_list = []
        for i in range(num_players):
            print(f'loop {i}')
            elo_list.append(pm.Normal(f'elo{i+1}', mu=1500, sigma=100))
        # Create lists of unobserved P RV's and observed game data RV's
        p_list = []
        data_list = []
        count = 0
        print('starting loops')
        for i in range(num_players):
            for j in range(i + 1, num_players):
                print(f'loop {i}, {j}')
                elo_diff = elo_list[j] - elo_list[i]
                p_list.append(1 / (1 + 10**(elo_diff/400)))
                # Check to see if there is data for this pairing
                if len(games[count]) > 0:
                    data_list.append(pm.Bernoulli(f'data_{i}{j}', p=p_list[-1], observed=games[count]))
                count += 1
        # Create Posterior Sample
        print('P LOOPS DONE')
        trace = pm.sample(return_inferencedata=False,draws=100, cores=1)
        print('SAMPLING DONE')
    for i in range(num_players):
        sample = trace.get_values(f'elo{i+1}')
        mean = sample.mean()
        std = sample.std()
        print(f'Driver {i+1} Elo: Mean = {mean}, Std = {std}')

def f1_game_list(num):
    """returns list of lists of 1 or 0 that comports with the game data required
    by the n_player_games function. Takes ndarray of drivers"""

    game_list = []
    num_players = num
    for i in range(num_players):
            for j in range(i + 1, num_players):
                game_list.append([1])
    return game_list


games_12 = [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
games_34 = [0, 0]
games_23 = [1, 0, 0, 0, 0, 0, 0, 0]

four_player_games = [[1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1], 
                     [], 
                     [],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [],
                     [0, 0]]
two_player_games = [[1, 0, 1, 1, 1, 0, 1]]

#n_players(20, f1_game_list(2))


import timeit

time_dict = {}
for i in range(2,10):
    starttime = timeit.default_timer()
    n_players(i, f1_game_list(i))
    time_dict[i] = timeit.default_timer() - starttime

for num in time_dict:
    print(f"Time for {num} players:", time_dict[num])



#n_players(4, f1_game_list)
#four_players(games_12, games_23, games_34)
        
    

