import sys, argparse, os, json
from src.metrics import compare_to_groundtruth
from src.data import Hicmat, load_hic_groundtruth, preprocess_file
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

    parser.add_argument(
        "--metrics_mode",
        default=False,
        action='store_true',
        help="If set, the program will produce the files with metrics inside",
    )

    parser.add_argument(
        "--gt_folder",
        default = None,
        help="Folder containing the ground truth files",
    )
    return parser.parse_args()

args = parse_arguments()
raw_path = os.path.join(args.folder, args.file)
data_path = os.path.splitext(raw_path)[0] + '.npy'

if args.resolution not in [25000, 100000]:
    sys.exit("BananaTAD support only resolution of 25kb or 100kb.")

assert not args.metrics_mode or args.gt_folder is not None, "If metrics_mode is set, gt_folder must be set" 

# If files weren't preprocessed, do it now
if not os.path.isfile(data_path):
    print(raw_path)
    preprocess_file(raw_path, args.resolution)

hic_mat = Hicmat(data_path, args.resolution, auto_filtering=True, cell_type=args.cell_type)
consensus_method = BordersConsensus(init=True)

if args.metrics_mode:
    hic_mat, arrowhead_tads = load_hic_groundtruth(data_path, 25000, arrowhead_folder=args.gt_folder)
    final_tads = consensus_method.get_consensus_tads(hic_mat=hic_mat) # TADs '(from, to)'
    savefile = open(os.path.join(hic_mat.get_folder(), hic_mat.get_name().replace('.npy', 'bananatads.txt')), 'w+')
    _, _, gt_rate_final, pred_rate_final = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=final_tads)
    savefile.write('TADs: {}\n'.format(final_tads))
    savefile.write('METRICS: (Ground Truth Rate: {}, Predicted Rate: {})\n'.format(gt_rate_final, pred_rate_final))
    savefile.close()
else:
    savefile = open(os.path.join(hic_mat.get_folder(), hic_mat.get_name().replace('.npy', '.bananatads')), 'w+')
    final_tads_scores = consensus_method.get_consensus(hic_mat=hic_mat) # TADs scores '(from, to):score'
    savefile.write('{}'.format(final_tads_scores))
    savefile.close()