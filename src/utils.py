import pandas as pd
import numpy as np

# load the available ArrowHead results
def read_arrowhead_result(path, chromosome, resolution):
    # TODO: Refactor
    df = pd.read_csv(path, sep='\t')
    df = df[df['chr1']==chromosome]
    list_tads = []
    for i in range(len(df)):
        list_tads.append((int(df['x1'][i]/resolution), int(df['x2'][i]/resolution)))
    return list_tads

def SCN(D, max_iter = 10):
	"""
	Input : spare array * int
	Out  : SCN(D)
	Code version from hictools.py, which was inspired by Boost-HiC paper
	"""    
	# Iteration over max_iter    
	for i in range(max_iter):
		D = np.divide(D, np.maximum(1, D.sum(axis = 0)))       
		D = np.divide(D, np.maximum(1, D.sum(axis = 1)))    
	return (D + D.T)/2 # To make matrix symetric again   