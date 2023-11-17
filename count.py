import numpy as np
import pandas as pd
import tifffile
import uuid
import matplotlib.pyplot as plt

class CountMatrix:
    def __init__(self,spots,mask):
        self.spots = spots
        self.mask = mask
    def assign_spots_to_cells(self,spots,mask):
        idx = np.round(spots[['x','y']].to_numpy()).astype(np.int16)
        labels = mask[idx[:,0],idx[:,1]]
        spots = spots.assign(label=labels)
        return spots
    def get_counts(self):
        self.spots = self.assign_spots_to_cells(self.spots,self.mask)
        self.spots = self.spots.loc[self.spots['label'] != 0]
        grouped = self.spots.groupby(['label'])
        count_mat = grouped.size().to_frame('counts')
        return count_mat           

  
