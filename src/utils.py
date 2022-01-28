import pandas as pd
import numpy as np
import os


def read_arrowhead_result(path, chromosome):
	"""
    Load the available ArrowHead results and put them in a tuples list
    ----------
    INPUT
    path : str
        path to the arrowhead results
    chromosome : str
        name of the chromosome to looking for
    -----------
    OUTPUT
    list_tads : list of tuples
		list of the TADs predicted by Arrowhead
    """
    # TODO: Refactor
    df = pd.read_csv(path, sep='\t', header=0)
    df = df[df['chr1']==chromosome]
    list_tads = []
    for i in df.index.tolist():
        list_tads.append((int(df['x1'][i]), int(df['x2'][i]))) # we get the TADs coordinates from the columns named "x1" and "x2"
    return list_tads

def SCN(D, max_iter = 10):
    """
    Improve the matrix display in the humain readable format
    Code version from hictools.py, which was inspired by Boost-HiC paper
    ----------
    INPUT
    D : npy matrix
        path to the arrowhead results
    max_iter : int
        maximum number of iterations
    -----------
    OUTPUT
    return a numpy matrix
    """
	for i in range(max_iter):
		D = np.divide(D, np.maximum(1, D.sum(axis = 0)))       
		D = np.divide(D, np.maximum(1, D.sum(axis = 1)))    
	return (D + D.T)/2 # To make matrix symetric again

def chrom_name_to_variables(chrom_name):
    """
    Put the name the cell type of the chromosome in two variables
    ----------
    INPUT
    chrom_name : str
        path to the chromosome
    -----------
    OUTPUT
    chrom : str
		name of the chromosome
    cell_type : str
        name of the cell type
    """
	len_path = len(chrom_name.split(os.path.sep))
	chrom = chrom_name.split(os.path.sep)[len_path-1][:4]
	if chrom[:3] != 'chr':
		raise ValueError('Chromosome name should start with "chr"')
	cell_type = None
	for ct in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'NHEK']:
		if ct in chrom_name:
			if not cell_type:
				cell_type = ct
			else: # check if only one cell type appears in the file name
				raise ValueError('Multiple cell types in the same file')
	if not cell_type:
		raise ValueError('No cell type found in the file')
	return chrom, cell_type
