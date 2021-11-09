
import numpy as np
import pandas as pd
import os, time, logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def load_hic(path, resolution):
    df = pd.read_csv(path, sep="\t",header=None, names=["i","j","score"])

    max_index = int(max(max(df.i)/resolution, max(df.j)/resolution))

    del df

    # Create square matrix
    matrix = np.zeros((max_index+1, max_index+1), dtype=int)

    def set_score(matrix, line):
        i,j,score = line.strip().split('\t')
        i = int(int(i)/resolution)
        j = int(int(j)/resolution)
        matrix[i,j] = int(float(score))
        matrix[j,i] = int(float(score))

    

    f = open(path, 'r')
    lines = f.readlines()    
    for line in lines:
        set_score(matrix, line)
    return matrix


def preprocess_data(folder, resolution):
    start_time = time.time()
    logging.basicConfig(filename="data.log", level=logging.DEBUG)
    logging.info('===================================================================================')
    logging.info('\tPreprocessing of folder {} started...'.format(folder))
    for f in os.listdir(folder):
        if f.endswith('.RAWobserved'):
            m = load_hic(os.path.join(folder, f), resolution=resolution)
            np.save(os.path.join(folder, f.replace(".RAWobserved",".npy")), m)
            logging.info('Preprocessing: file {} preprocessed'.format(f))
        else:
            logging.info('Preprocessing: file {} skipped'.format(f))
    logging.info("Preprocessing finished after {} seconds".format(time.time() - start_time))

# TODO: Move it to a notebook
def plot_data(m, region=None, scale='log'):
    if scale == 'log':
        m = np.log(m)
    if type(region) is tuple:
        # Subset of the file - zoom on a region
        m = m[region[0]:region[1], region[0]:region[1]]
        Vmax = m.max()/np.log10((region[1]-region[0])/10)
    else:
        Vmax = m.max()/np.log10(len(m)/10)
    fig, ax = plt.subplots()
    shw = ax.imshow(m, cmap='OrRd', vmin=0, vmax=Vmax, interpolation ='none', 
              origin ='upper')
    bar = plt.colorbar(shw)
    bar.set_label('Scale')
    plt.show()

class Hicmat:
    def __init__(self, path, resolution):
        m = np.load(path)
        self.resolution = resolution
        self.original_matrix = m
        self.filter_coords = None
        self.reduced_matrix = None

    def filter(self, threshold = 0):
        if self.filter_coords is not None or self.reduced_matrix is not None:
            logging.info('Matrix already filtered')
            return
        sum_row_col = self.original_matrix.sum(axis=0) + self.original_matrix.sum(axis=1)
        reduced_idx = np.where(sum_row_col <= threshold)
        self.reduced_matrix = self.original_matrix.copy()
        self.reduced_matrix = np.delete(self.reduced_matrix, reduced_idx, axis=0)
        self.reduced_matrix = np.delete(self.reduced_matrix, reduced_idx, axis=1)

