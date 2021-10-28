
import numpy as np
import pandas as pd
import os, time
from joblib import Parallel, delayed

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
    # TODO: Add logging
    start_time = time.time()
    for line in lines:
        set_score(matrix, line)
    print(time.time() - start_time, "seconds")


