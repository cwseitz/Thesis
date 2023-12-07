import matplotlib.pyplot as plt
from figures import Figure_4
import json

prefix = '230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-1_spots'

with open('plot_figure_4.json', 'r') as f:
    config = json.load(f)
    
figure = Figure_4(config,prefix)
figure.plot()
plt.show()
