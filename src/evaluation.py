import contextlib, os, io, logging, platform
import numpy as np
from tqdm import tqdm

from src.consensus import BordersConsensus
from src.data import load_hic_groundtruth
from src.metrics import compare_to_groundtruth


def evaluate_set(test_set):
    """
    Evaluate the performance of the model on the test set
    ----------
    INPUT
    test_set : list of str
        list of file paths containing chromosomes of the testing dataset
    -----------
    OUTPUT
    a file for each chromosome in the testing dataset containing the predicted TADs and the results of the 
    both metrics that compare predictions with the ground truth TADs
    """
    set_25kb = []
    set_100kb = []
    for f in test_set:
        if '25kb' in f:
            set_25kb.append(f)
        elif '100kb' in f:
            set_100kb.append(f)
        else:
            raise ValueError('File name {} was unexpected'.format(f))
    consensus_method = BordersConsensus(init=True)

    # Find the TADs from the consensus on each TADs with a resolution of 25kb and compare them to the ground truth TADs
    for f_25kb in tqdm(set_25kb):
        logging.info('Evaluating {}'.format(f_25kb))
        savefile = open(f_25kb.replace('npy', 'bananatads.txt'), 'w+')
        hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
        # find the consensus
        consensus_tads = consensus_method.get_consensus_tads(hic_mat=hic_mat)
        # evaluate the consensus with the ground truth
        _, _, gt_rate_consensus, pred_rate_consensus = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=consensus_tads, gap=200000)
        # save results in a file
        savefile.write('TADs: {}\n'.format(consensus_tads))
        savefile.write('METRICS: (Ground Truth Rate: {}, Predicted Rate: {})\n'.format(gt_rate_consensus, pred_rate_consensus))
        savefile.close()
        results_name = os.path.join('results', '')
        for s in f_25kb.replace('npy', 'bananatads.txt').split(os.path.sep)[2:]:
            results_name += s
            if not s.endswith('.bananatads.txt'):
                results_name += '_'
        if platform.system() == 'Windows':
            os.system(f"copy {f_25kb.replace('npy', 'bananatads.txt')} {results_name}")
        else:
            os.system(f"cp {f_25kb.replace('npy', 'bananatads.txt')} {results_name}")
    # Find the TADs from the consensus on each TADs with a resolution of 100kb and compare them to the ground truth TADs
    for f_100kb in tqdm(set_100kb):
        logging.info('Evaluating {}'.format(f_100kb))
        savefile = open(f_100kb.replace('npy', 'bananatads.txt'), 'w+')
        hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
        # find the consensus
        consensus_tads = consensus_method.get_consensus_tads(hic_mat=hic_mat)
        # evaluate the consensus with the ground truth
        _, _, gt_rate_consensus, pred_rate_consensus = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=consensus_tads, gap=200000)
        # save results in a file
        savefile.write('TADs: {}\n'.format(consensus_tads))
        savefile.write('METRICS: (Ground Truth Rate: {}, Predicted Rate: {})\n'.format(gt_rate_consensus, pred_rate_consensus))
        savefile.close()
        results_name = os.path.join('results', '')
        for s in f_100kb.replace('npy', 'bananatads.txt').split(os.path.sep)[2:]:
            results_name += s
            if not s.endswith('.bananatads.txt'):
                results_name += '_'
        if platform.system() == 'Windows':
            os.popen(f"copy {f_100kb.replace('npy', 'bananatads.txt')} {results_name}")
        else:
            os.popen(f"cp {f_100kb.replace('npy', 'bananatads.txt')} {results_name}")
    print('\Evaluation finished!')

def save_split(development_set, test_set):
    """
    Put the split of all the dataset in a file
    ----------
    INPUT
    development_set : list of str
        list of file paths containing chromosomes of the trainig dataset
    test_set : list of str
        list of file paths containing chromosomes of the testing dataset
    -----------
    OUTPUT
    a file containing the name of all the chromosomes of the training dataset
    a file containing the name of all the chromosomes of the testing dataset
    """
    trainfile = open(os.path.join('results', 'train_split.txt'), 'w+')
    trainfile.write('TRAIN SET:\n{}'.format(development_set))
    trainfile.close()
    testfile = open(os.path.join('results', 'test_split.txt'), 'w+')
    testfile.write('TEST SET:\n{}'.format(test_set))
    testfile.close()