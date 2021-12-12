
import numpy as np
import pandas as pd
import os, time, logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
def plot_data(m, region=None, scale='log', tads=None, resolution=None):
    if scale == 'log':
        m = np.log(m)
    if type(region) is tuple:
        # Subset of the file - zoom on a region
        if resolution is None:
            raise ValueError("Resolution must be specified for zoomed plots")
        start = int(region[0]/resolution)
        end = int(region[1]/resolution)
        m = m[start:end, start:end]
        Vmax = m.max()/np.log10(len(m)/10)
    else:
        Vmax = m.max()/np.log10(len(m)/10)
    fig, ax = plt.subplots()
    shw = ax.imshow(m, cmap='OrRd', vmin=0, vmax=Vmax, interpolation ='none', 
              origin ='upper')
    
    bar = plt.colorbar(shw)
    bar.set_label('Scale')
    if tads is not None:
        if resolution is None:
            raise Exception('Resolution is not specified')
        for tad in tads:
            tad_length = (tad[1]-tad[0]) / resolution
            xy = (int(tad[0]/resolution), int(tad[0]/resolution))
            ax.add_patch(Rectangle(xy, tad_length, tad_length, fill=False, edgecolor='blue', linewidth=1))
    plt.show()

class Hicmat:
    def __init__(self, path, resolution):
        m = np.load(path)
        if m.shape[0] != m.shape[1]:
            raise ValueError('Matrix is not square')
        self.resolution = resolution
        self.original_matrix = m
        self.filtered_coords = None
        self.reduced_matrix = None
        self.regions = None

    def filter(self, threshold = 0, min_length_region=5): # TODO: Discuss about min_length_region
        if self.filtered_coords is not None or self.reduced_matrix is not None:
            logging.info('Matrix already filtered')
            return
        sum_row_col = self.original_matrix.sum(axis=0) + self.original_matrix.sum(axis=1)
        self.filtered_coords = np.where(sum_row_col <= (threshold*(self.original_matrix.shape[0]+self.original_matrix.shape[1])) )[0]

        self.regions = []
        for i in range(len(self.filtered_coords)-1):
            # Save indexes of regions
            if self.filtered_coords[i+1] - self.filtered_coords[i] >= min_length_region:
                self.regions.append((self.filtered_coords[i], self.filtered_coords[i+1]))

        self.reduced_matrix = self.original_matrix.copy()
        self.reduced_matrix = np.delete(self.reduced_matrix, self.filtered_coords, axis=0)
        self.reduced_matrix = np.delete(self.reduced_matrix, self.filtered_coords, axis=1)

    def get_regions(self):
        if self.regions is None:
            raise ValueError('Matrix not filtered')
        return self.regions

