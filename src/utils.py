import pandas as pd
import numpy as np
import os

# load the available ArrowHead results
def read_arrowhead_result(path, chromosome):
    # TODO: Refactor
    df = pd.read_csv(path, sep='\t', header=0)
    df = df[df['chr1']==chromosome]
    list_tads = []
    for i in df.index.tolist():
        list_tads.append((int(df['x1'][i]), int(df['x2'][i])))
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

def chrom_name_to_variables(chrom_name):
	chrom = chrom_name[:4]
	cell_type = None
	for ct in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'NHEK']:
		if ct in chrom_name:
			if not cell_type:
				cell_type = ct
			else:
				raise ValueError('Multiple cell types in the same file')
	if not cell_type:
		raise ValueError('No cell type found in the file')
	return chrom, cell_type
