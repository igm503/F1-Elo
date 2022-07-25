from turtle import color
import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm

def elo_results(num_wins, num_games, elos):
    return results_elo(num_wins, num_games, elos) * np.product(prior_elo(elos), axis=elos.ndim-1)

def results_elo(num_wins, num_games, elos):
    """probability of player 1 with elo1 winning num_wins games out of num_games games"""
    print('doing results_p(elos)')
    prob = p(elos)
    print('doing results_elo')
    return np.product(comb(num_games, num_wins) * prob**num_wins * (1 - prob)**(num_games - num_wins), axis=elos.ndim-1)

prior_elo = lambda x: norm.pdf(x, loc=1500, scale=100)

def p(elos):
    """ elos is an nd array with n = # of players. 
    
    probability player with elo1 beats player with elo2. If elo1 and elo2 are arrays, returns 2d array 
    where element [i, j] is the probability that player with elo1[i] beat player with elo2[j] """
    num_players = elos.ndim - 1
    num_comparisons = num_players * (num_players - 1) / 2
    elo_diff_shape = list(elos.shape)
    elo_diff_shape[-1] = int(num_comparisons)
    elo_diff = np.ones(elo_diff_shape)

    count = 0
    for i in tqdm(range(num_players)):
        for j in range(i):
            elo_diff[..., count] = elos[...,j] - elos[...,i]
            count += 1
    return 1 / (1 + 10**(elo_diff/400))

def margin(num_wins, num_games, elo, player):
    """elo is a ndarray of elo values you want the marginal probability for. Returns ndarray of marginal probabilities 
    for those elo values given num_wins and num_games. Axis specifies which player you want. player=1 specifies player 1,
    player=2 specifies player 2"""
    if player == 1:
        joint_prob = elo_results(num_wins, num_games, elo, np.arange(0, 10000))
        ax = 1
    if player == 2:
        joint_prob = elo_results(num_wins, num_games, np.arange(0, 10000), elo)
        ax = 0
    norm_constant = np.sum(joint_prob)
    return np.sum(joint_prob, axis=ax) / norm_constant

def n_elo_space(n, lower, upper):
    elo_line = np.arange(lower, upper, 10)
    dimensions = []
    for i in range(n):
        dimensions.append(elo_line)
    print('here we go')
    return np.array(np.meshgrid(*dimensions)).T


num_wins = np.array([2, 0, 0, 1, 0, 2])
num_games = np.array([2, 0, 0, 2, 0, 2])
elos = n_elo_space(4, 1000, 2000)
print('did it')
plt.plot(np.arange(1000, 2000, 50), np.sum(elo_results(num_wins, num_games, elos), axis= (1,2,3)), color='red', label='Player 1')
plt.plot(np.arange(1000, 2000, 50), np.sum(elo_results(num_wins, num_games, elos), axis= (0,2,3)), label='Player 2') 
plt.plot(np.arange(1000, 2000, 50), np.sum(elo_results(num_wins, num_games, elos), axis= (1,0,3)), color='blue', label='Player 3') 
plt.plot(np.arange(1000, 2000, 50), np.sum(elo_results(num_wins, num_games, elos), axis= (0,1,2)), label='Player 4')

plt.legend()
plt.show()



