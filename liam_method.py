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
        np.savetxt(os.path.join(folder, f.replace(".RAWobserved.txt",".txt")), m, delimiter=' ', fmt='%d')
        logging.info('Preprocessing: file {} preprocessed'.format(f))
    logging.info("Preprocessing finished after {} seconds".format(time.time() - start_time))


# TODO: Move it to a notebook
def plot_data(path, region=None, scale='log'):
    m = np.load(path)
    fig, ax = plt.subplots()
    if scale == 'log':
        m = np.log(m)
    Vmax = m.max()/np.log10(len(m)/10)
    if type(region) is tuple:
        # Subset of the file - zoom on a region
        m = m[region[0]:region[1]+1, region[0]:region[1]+1]
        Vmax = m.max()/np.log10((region[1]-region[0])/10)
    ax.imshow(m, cmap='jet', vmin=0, vmax=min(Vmax, m.max()), interpolation ='none', 
              origin ='lower')
    plt.show()

def found_TADs(path, window):
    mat = np.log(np.load(path))
    mat[mat=='-inf'] = 0
    
    def get_TADs(mat):
        list_TADs = []
        n = len(mat)
        tad = False
        last_tad = -1
        for i in range(n-window+1):
            square = mat[i:i+window, i:i+window]
            tad_square = square.copy()
            k = 0
            while np.quantile(tad_square, 0.2)>3 and not tad and i+k<n-window:
            #while np.mean(tad_square)>3:
                tad_square = mat[i:i+window+k, i:i+window+k]
                k+=1
            if np.quantile(square, 0.2)>2 and not tad:
            #if np.mean(square)>3 and not tad:
                list_TADs.append((i, i+k+window))
                tad = True
                last_tad = i+k-1
            elif i>last_tad :
                tad = False
        return list_TADs
    
    TADs_reverse = []
    for tad in get_TADs(mat[::-1, ::-1]):
        TADs_reverse.append((len(mat)-tad[1], len(mat)-tad[0]))
    list_all_TADs = get_TADs(mat) + TADs_reverse
    
    return mat, list_all_TADs

def display_TADs(path, list_TADs):
    mat = np.log(np.load(path))
    fig, ax = plt.subplots()
    Vmax = mat.max()/np.log10(len(mat)/10)
    ax.imshow(mat, cmap='jet', vmin=0, vmax=min(Vmax, mat.max()), interpolation ='none', 
              origin ='lower')
    for tad in list_TADs:
        ax.add_patch(patches.Rectangle((tad[0], tad[0]), 
                                       tad[1]-tad[0], 
                                       tad[1]-tad[0], 
                                       fill=False,
                                       edgecolor='black'))
    plt.show()
