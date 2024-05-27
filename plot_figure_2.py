import matplotlib.pyplot as plt
from figures import Figure_2
import json

prefixes = [
'230722_QD-Cal-HighNA-Astigmatism',
'230722_QD-Cal-LowNA-Astigmatism',
'230722_QD-Cal-HighNA-Isotropic',
'230722_QD-Cal-LowNA-Isotropic'
]

with open('plot_figure_2.json', 'r') as f:
    config = json.load(f)

figure = Figure_2(config)
figure.plot(prefixes)

