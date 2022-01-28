#import numpy as np
#import pandas as pd
import sys, argparse, os#, time, logging
from src.data import Hicmat, preprocess_data #, HiCDataset, plot_data, load_hic_groundtruth
#from src.tad_algo import TopDom, TADtree, OnTAD, TADbit
#from src.metrics import compare_to_groundtruth
from src.consensus import BordersConsensus
from src.utils import *
 
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This program products output files containing TADs found by different algorithms which are necessary to do the consensus of BananaTAD"
    )
    parser.add_argument(
        "--file",
        help="Path to a file containing a contact matrix.",
    )
    parser.add_argument(
        "--folder",
        help="Folder containing the chromosome(s) to preprocess potentially.",
    )
    parser.add_argument(
        "--cell_type",
        default=None,
        help="cell type of the chromosome (ex: GM12878",
    )
    parser.add_argument(
        "--chrom",
        default=None,
        help="name of the chromosome",
    )
    parser.add_argument(
        "--resolution",
        default=100000,
        type=int,
        help="Resolution of the HiC data",
    )
    return parser.parse_args()

args = parse_arguments()
data_path = os.path.join(args.folder, args.file)

if args.resolution not in [25000, 100000]:
    sys.exit("BananaTAD support only resolution of 25kb or 100kb.")

# If files weren't preprocessed, do it now
if not os.path.isfile(data_path):
    preprocess_data(args.folder, args.resolution)

hic_mat = Hicmat(data_path, args.resolution, auto_filtering=True, cell_type=args.cell_type)
consensus_method = BordersConsensus(init=True)
final_tads = consensus_method.get_consensus_tads(hic_mat=hic_mat, threshold=10) # TADs '(from, to)'
final_tads_scores = consensus_method.get_consensus(hic_mat=hic_mat, threshold=10) # TADs scores '(from, to):score'