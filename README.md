# F1 Elo

This is a WIP attempt to rate F1 drivers against each other using an Elo system. One file uses classic k-factor updating for each race, while the other uses a bayesian update rule. The bayesian solution has an option to give cars their own Elo ratings. Currently, this doesn't seem to be working, as the cars are given a much smaller role in determining performance than they probably ought to be (e.g. the Mercedes W-11 is estimated to add less than 100 Elo points to L. Hamilton's > 1500 Elo points).

The data I use to make these ratings comes from www.racing-reference.info. The file htmlscraper.py is a simple html scraper that grabs finish position data from the tables on that website. 
