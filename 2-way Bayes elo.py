import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def elo_results(num_wins, num_games, elo1, elo2):
    return results_elo(num_wins, num_games, elo1, elo2) * prior_elo1(elo1) * prior_elo2(elo2)

def results_elo(num_wins, num_games, elo1, elo2):
    """probability of player 1 with elo1 winning num_wins games out of num_games games"""
    prob = p(elo1, elo2)
    return comb(num_games, num_wins) * prob**num_wins * (1 - prob)**(num_games - num_wins)

prior_elo1 = lambda x: norm.pdf(x, loc=1500, scale=100)
prior_elo2 = lambda x: norm.pdf(x, loc=1500, scale=100)

def p(elo1, elo2):
    """probability player with elo1 beats player with elo2. If elo1 and elo2 are arrays, returns 2d array 
    where element [i, j] is the probability that player with elo1[i] beat player with elo2[j] """
    if isinstance(elo2, np.ndarray):
        elo_diff = elo2 - elo1.reshape(elo1.size,1)
    else:
        elo_diff = elo2 - elo1
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

     

elo1 = np.arange(0, 10000)
elo2 = np.arange(0, 3000)


#plt.plot(elo1, elo_results(10, 10, elo1, 1500))
#plt.plot(elo1, results_elo(10, 10, elo1, 1500))

#plt.plot(elo1, margin(1, 1, elo1, 1))
#plt.plot(elo1, margin(1, 1, elo1, 2))

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = np.arange(0, 3000)
yline = np.arange(0, 3000)
zline = elo_results(0, 0, xline, yline)
print(zline.shape)
ax.contour3D(xline, yline, zline, 50, cmap='binary')
ax.set_xlabel('elo1')
ax.set_ylabel('elo2')
ax.set_zlabel('prob')


plt.show()



