import numpy as np
import matplotlib.pyplot as plt
import tifffile
from SMLM.tests import *
import pybisp as pb
import pandas as pd

class Figure_7:
    """Diffusion dynamics"""
    def __init__(self,config,prefix):
        self.config = config
        self.prefix = prefix
        self.spots = pd.read_csv(config['analpath'] + prefix + '.csv')

    def bayes(self,dt=0.01,k_B=1.38064881313131e-17,T=300,min_traj_length=80):
        spots = self.spots
        particles = spots['particle'].unique()
        k_vec = []; D_vec = []
        for particle in particles:
            traj_length = len(spots.loc[spots['particle'] == particle])
            if traj_length > min_traj_length:
                this_spots = spots.loc[spots['particle'] == particle]
                x = this_spots['x_mle']; y = this_spots['y_mle']
                plt.plot(x); plt.show()
                
                ou = pb.ou.Inference(this_spots['x'].values, dt)
                L, D, K = ou.mapEstimate() 
                k_ou = K * (k_B*T) # physical value of the stiffness (N /micron)
                print('The best estimate for the stiffness is', k_ou, 'N/muM')
                print(D)
                
                eqp = pb.equipartition.Inference(this_spots['x'].values)
                K = eqp.mapEstimate()
                k_eq = K * (k_B*T) # physical value of the stiffness (N /micron)
                print('The best estimate for the stiffness is', k_eq, 'N/muM ')
                
                k_vec.append([k_ou,k_eq])
                
        k_vec = np.array(k_vec)
        plt.scatter(k_vec[:,0],k_vec[:,1],color='black')
        plt.show()
                

    def plot(self):
        fig, ax = plt.subplots(figsize=(6,6))
        self.bayes()

        

