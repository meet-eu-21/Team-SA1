import numpy as np
import pandas as pd
import os, time, logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# load an HiC file with his resolution and return a matrix in numpy format
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

# preprocess all the Hic files contained in a folder (with the same resolution)
def preprocess_data(folder, resolution):
    start_time = time.time()
    logging.basicConfig(filename="data.log", level=logging.DEBUG)
    logging.info('===================================================================================')
    logging.info('\tPreprocessing of folder {} started...'.format(folder))
    for f in os.listdir(folder):
        m = load_hic(os.path.join(folder, f), resolution)
        # put the matrix in a numpy file
        np.save(os.path.join(folder, f.replace(".RAWobserved.txt",".npy")), m)
        # put the matrix in a .txt (for TADtree)
        np.savetxt(os.path.join(folder, f.replace(".RAWobserved.txt",".txt")), m, delimiter=' ', fmt='%d')
        logging.info('Preprocessing: file {} preprocessed'.format(f))
    logging.info("Preprocessing finished after {} seconds".format(time.time() - start_time))

# plot a contact map of an HiC file, possibility to zoom on a zone and to delimite it 
def plot_data(path, region=None, scale='log'):
    m = np.load(path)
    fig, ax = plt.subplots()
    if scale == 'log':
        m = np.log(m)
    # scale of color if we don't zoom
    Vmax = m.max()/(len(m)/1500)
    # case of zooming
    if type(region) is tuple:
        # find position of the rectangle which will delemite the zone
        if max(0, region[0]-20)==0:
            start = region[0]
            end = region[1]+21-region[0]
        elif min(len(m), region[1]+21)==len(m):
            start = 20
            end = len(m)-region[0]
        else:
            start = 20
            end = region[1]+21-region[0]
        # part of the matrix which will be shown in the contact map
        m = m[max(0, region[0]-20):min(len(m), region[1]+21), max(0, region[0]-20):min(len(m), region[1]+21)]
        # readjust scale of color
        Vmax = m.max()
    # display the contact map
    ax.imshow(m, cmap='YlOrRd', vmin=0, vmax=min(Vmax, m.max()), interpolation ='none', 
              origin ='lower')
    # display the rectangle
    if type(region) is tuple:
        ax.add_patch(patches.Rectangle((start, start), 
                                        end-start, 
                                        end-start, 
                                        fill=False,
                                        edgecolor='green'))
    plt.show()

# arbitrary method to find TADs of an HiC file, size of sliding windows necessary (=size minimum of a TAD)
def found_TADs(path, window):
    mat = np.log(np.load(path))
    mat[mat=='-inf'] = 0
    # method to find TADs in a sens of lecture (no overlaps)
    def get_TADs(mat, window):
        list_TADs = []
        n = len(mat)
        tad = False
        last_tad = -1
        for i in range(n-window+1):
            square = mat[i:i+window, i:i+window]
            tad_square = square.copy()
            k = 0
            while np.quantile(tad_square, 0.2)>3 and not tad and i+k<n-window:
                tad_square = mat[i:i+window+k, i:i+window+k]
                k+=1
            # to enlarge the TAD size
            if np.quantile(square, 0.2)>2 and not tad:
                list_TADs.append((i, i+k+window))
                tad = True
                last_tad = i+k-1
            elif i>last_tad :
                tad = False
        return list_TADs
    # apply 'get_TADs' in both directions
    TADs_reverse = []
    for tad in get_TADs(mat[::-1, ::-1], window):
        TADs_reverse.append((len(mat)-tad[1], len(mat)-tad[0]))
    list_all_TADs = get_TADs(mat, window) + TADs_reverse
    return mat, list_all_TADs

# display the contact map with the associated TADs delimited
def display_TADs(path, list_TADs):
    mat = np.log(np.load(path))
    fig, ax = plt.subplots()
    print(mat.max())
    Vmax = mat.max()/(len(mat)/1500)
    # display the contact map
    ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=min(Vmax, mat.max()), interpolation ='none', 
              origin ='lower')
    # display a rectangle for each TAD
    for tad in list_TADs:
        ax.add_patch(patches.Rectangle((tad[0], tad[0]), 
                                       tad[1]-tad[0], 
                                       tad[1]-tad[0], 
                                       fill=False,
                                       edgecolor='green'))
    plt.show()

# load the available ArrowHead results
def read_arrowhead_result(path, chromosome, resolution):
    df = pd.read_csv(path, sep='\t')
    df = df[df['chr1']==chromosome]
    list_tads = []
    for i in range(len(df)):
        list_tads.append((int(df['x1'][i]/resolution), int(df['x2'][i]/resolution)))
    return list_tads