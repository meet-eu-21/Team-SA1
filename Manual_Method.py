import numpy as np
import pandas as pd
import os, time, logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_hic(path, resolution):
    # Get the number of rows and columns of the matrix
    df = pd.read_csv(path, sep="\t",header=None, names=["i","j","score"])
    max_index = int(max(max(df.i)/resolution, max(df.j)/resolution))
    del df

    # Create square matrix
    matrix = np.zeros((max_index+1, max_index+1), dtype=int)
    matrix_bis = np.zeros((max_index+1, max_index+1), dtype=int)
    
    # function that puts a value twice in the matrix
    def set_score(matrix, line):
        i,j,score = line.strip().split('\t')
        i = int(int(i)/resolution)
        j = int(int(j)/resolution)
        matrix[i,j] = int(float(score))
        matrix[j,i] = int(float(score))

    # fill the matrix
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
        m = load_hic(os.path.join(folder, f), resolution)
        np.save(os.path.join(folder, f.replace(".RAWobserved.txt",".npy")), m)
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
              origin ='lower')
    plt.show()

def found_TADs(path, window):
    mat = np.log(np.load(path))
    mat[mat=='-inf'] = 0 # replace missing values by 0
    n = len(mat)
    list_TADs = []
    tad = False
    last_tad = -1
    for i in range(n-window+1):
        square = mat[i:i+window, i:i+window]
        tad_square = square.copy()
        k = 0
        while np.mean(tad_square)>3: # condition to be a TAD (to change)
            # if the condition is validated we extend the square
            tad_square = mat[i:i+window+k, i:i+window+k]
            k+=1
        if np.mean(square)>3 and not tad: # second condition prevent to overlapped TADs
            list_TADs.append((i, i+k+window))
            tad = True
            last_tad = i+k-1
        elif i>last_tad :
            tad = False
    
    return list_TADs

def display_TADs(path, list_TADs):
    mat = np.log(np.load(path))
    fig, ax = plt.subplots()
    ax.imshow(mat, cmap='OrRd', vmin=0, vmax=mat.max(), interpolation ='none', 
              origin ='lower')
    # we display the TADs on the matrix
    for tad in list_TADs:
        ax.add_patch(patches.Rectangle((tad[0], tad[0]), 
                                       tad[1]-tad[0], 
                                       tad[1]-tad[0], 
                                       fill=False,
                                       edgecolor='red'))
    plt.show()
