#import numpy as np
#import pandas as pd
import sys #, os, time, logging
from src.data import Hicmat, preprocess_data #, HiCDataset, plot_data, load_hic_groundtruth
#from src.tad_algo import TopDom, TADtree, OnTAD, TADbit
#from src.metrics import compare_to_groundtruth
from src.consensus import BordersConsensus
from src.utils import *
 
if len(sys.argv)> 4:
    raise ValueError('Too much parameters')
data_path = sys.argv[1]
if sys.argv[3]:
    resolution = sys.argv[2]
else:
    resolution = 100000
cell_type = sys.argv[3]

# If files weren't preprocessed, do it now
#if not os.path.isfile(data_path):
#    preprocess_data(folder, resolution)

hic_mat = Hicmat(data_path, resolution, auto_filtering=True, cell_type=cell_type)
consensus_method = BordersConsensus(init=True)
final_tads = consensus_method.get_consensus_tads(hic_mat=hic_mat, threshold=10) # TADs '(from, to)'
final_tads_scores = consensus_method.get_consensus(hic_mat=hic_mat, threshold=10) # TADs scores '(from, to):score'