
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
    matrix_bis = np.zeros((max_index+1, max_index+1), dtype=int)

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
        m = load_hic(os.path.join(folder, f), resolution=resolution)
        np.save(os.path.join(folder, f.replace(".RAWobserved",".npy")), m)
        logging.info('Preprocessing: file {} preprocessed'.format(f))
    logging.info("Preprocessing finished after {} seconds".format(time.time() - start_time))

# TODO: Move it to a notebook
def plot_data(path, region, scale='log'):
    m = np.load(path)
    if type(region) is tuple:
        # Subset of the file - zoom on a region
        m = m[region[0]:region[1], region[0]:region[1]]
    fig, ax = plt.subplots()
    if scale == 'log':
        m = np.log(m)
    ax.imshow(m, cmap='OrRd', vmin=0, vmax=m.max(), interpolation ='none', 
              origin ='upper')
    plt.show()
