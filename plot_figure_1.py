import matplotlib.pyplot as plt
from figures import Figure_1
import json
import pandas as pd

prefix = 'spots'
with open('plot_figure_1.json', 'r') as f:
    config = json.load(f)
    
spots = pd.read_csv(config['path'] + prefix + '.csv')
spots = spots.loc[spots['uncertainty [nm]'] < 50.0]
spots['x'] = spots['x [nm]']/108.3
spots['y'] = spots['y [nm]']/108.3
figure = Figure_1(spots)
figure.plot()
plt.show()
