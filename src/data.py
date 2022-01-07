
import numpy as np
import pandas as pd
import os, time, logging, subprocess, platform
import random
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from src.utils import SCN, read_arrowhead_result

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
        if f.endswith('.RAWobserved'):
            m = load_hic(os.path.join(folder, f), resolution=resolution)
            # put the matrix in a numpy file
            np.save(os.path.join(folder, f.replace(".RAWobserved",".npy")), m)
            # put the matrix in a .txt (for TADtree)
            np.savetxt(os.path.join(folder, f.replace(".RAWobserved",".txt")), m, delimiter=' ', fmt='%d')
            logging.info('Preprocessing: file {} preprocessed'.format(f))
        else:
            logging.info('Preprocessing: file {} skipped'.format(f))
    logging.info("Preprocessing finished after {} seconds".format(time.time() - start_time))

# plot a contact map of an HiC file, possibility to zoom on a zone and to delimite it 
def plot_data(m, resolution, region=None, scale='log', tads=None):
    original_len = len(m)
    if scale == 'log':
        m = np.log10(m)
        m = SCN(m)
        # Vmax = m.max()/np.log10(len(m)/10)
        Vmax = m.max()
        Vmin = m.min()
        # TODO: find something for contrast diagonal / other
    if type(region) is tuple:
        # Subset of the file - zoom on a region
        if resolution is None:
            raise ValueError("Resolution must be specified for zoomed plots")
        # dezoom a bit to highlight the region
        region_length = (region[1]-region[0])/resolution
        dezoom = int(region_length/20)
        # TODO: Check
        start = max(int((region[0]/resolution)-dezoom),0)
        end = min(int((region[1]/resolution)+dezoom),original_len-1)
        m = m[start:end, start:end]
        # Vmax = m.max()/np.log10(len(m)/10)
    # else:
        # Vmax = m.max()/np.log10(len(m)/10)

    fig, ax = plt.subplots()
    shw = ax.imshow(m, cmap='OrRd', interpolation ='none', 
              origin ='upper')
    
    if type(region) is tuple:
        start_idx = max(int(region[0]-dezoom),0)
    else:
        start_idx = 0
    xticks, _ = plt.xticks()
    xticks_cor = xticks[1:-1]
    yticks, _ = plt.yticks()
    yticks_cor = yticks[1:-1]
    plt.xticks(ticks=xticks_cor, labels=['{}'.format(int(((b+start_idx)*resolution)/1000000)) for b in xticks_cor])
    plt.yticks(ticks=yticks_cor, labels=['{}'.format(int(((b+start_idx)*resolution)/1000000)) for b in yticks_cor])
    bar = plt.colorbar(shw)
    bar.set_label('Scale')
    if tads is not None:
        for tad in tads:
            tad_length = (tad[1]-tad[0]) / resolution
            xy = (int(tad[0]/resolution), int(tad[0]/resolution))
            ax.add_patch(Rectangle(xy, tad_length, tad_length, fill=False, edgecolor='blue', linewidth=1))
    if region is not None:
        dezoom_left = min(start, dezoom)
        dezoom_right = min(original_len-end, dezoom)
        ax.add_patch(Rectangle((dezoom_left, dezoom_left), 
                                len(m)-(dezoom_left+dezoom_right),
                                len(m)-(dezoom_left+dezoom_right), 
                                fill=False,
                                edgecolor='black',
                                linewidth=2))
    plt.show()

class Hicmat:
    def __init__(self, path, resolution):
        m = np.load(path)
        if m.shape[0] != m.shape[1]:
            raise ValueError('Matrix is not square')
        self.resolution = resolution
        self.original_matrix = np.array(m)
        self.filtered_coords = None
        self.reduced_matrix = None
        self.regions = None
        self.path = path

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

    def get_folder(self):
        return os.path.dirname(self.path)

    def get_name(self):
        return os.path.basename(self.path)



def load_hic_groundtruth(data_path, resolution, arrowhead_folder=os.path.join('data', 'TADs'), threshold=1):
	"""
		Function to correctly load the HiC matrix and its corresponding Arrowhead results
	"""
	path_comps = data_path.split(os.sep)

	if 'GM12878' in path_comps:
		# GM12878
		chr = path_comps[path_comps.index('GM12878')+2].split('_')[0][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'HMEC' in path_comps:
		# HMEC
		chr = path_comps[path_comps.index('HMEC')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_HMEC_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'HUVEC' in path_comps:
		# HUVEC
		chr = path_comps[path_comps.index('HUVEC')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_HUVEC_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'IMR90' in path_comps:
		# IMR90
		chr = path_comps[path_comps.index('IMR90')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_IMR90_Arrowhead_domainlist.txt'), chromosome=chr)
	elif 'NHEK' in path_comps:
		# NHEK
		chr = path_comps[path_comps.index('NHEK')+2][3:]
		arrowhead_tads = read_arrowhead_result(os.path.join(arrowhead_folder, 'GSE63525_NHEK_Arrowhead_domainlist.txt'), chromosome=chr)
	
	hic_mat = Hicmat(data_path, resolution)
	hic_mat.filter(threshold = threshold)

	return hic_mat, arrowhead_tads
