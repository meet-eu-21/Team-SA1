#!/bin/bash

python - <<END
import platform, logging
print('Init tuning from platform {}'.format(platform.system()))
from src.data import HiCDataset
import src.utils
import src.metrics
import src.tad_algo
from src.tuning import tune_topdom, tune_tadtree, tune_ontad, tune_tadbit, tune_borders_consensus_threshold, precompute_all_data
logging.basicConfig(filename='dev.log', filemode='a+', level=logging.INFO)
logging.info('Start tuning')

print('Init dataset')
dataset = HiCDataset(data_folder='data')
dataset.build_data_dict()
development_set, test_set = dataset.split(dev_ratio = 0.7, test_ratio=0.3)
# precompute_all_data(development_set)
# precompute_all_data(test_set)

# print('\nTuning TopDom...')
# print('\tParameter window')
# tune_topdom(development_set, param_ranges={'window': (1,15)})

# print('\nTuning TADTree...')
# print('\tParameter N')
# tune_tadtree(development_set, param_ranges={'N': (100,600)})
# print('\tParameter p')
# tune_tadtree(development_set, param_ranges={'p': (2,5)})

# print('\nTuning OnTAD...')
# print('\tParameter penalty')
# tune_ontad(development_set, param_ranges={'penalty': (0.05,0.35)})
# print('\tParameter log2')
# tune_ontad(development_set, param_ranges={'log2': (True,False)})

# print('\nTuning TADbit...')
# print('\tParameter score_threshold')
# tune_tadbit(development_set)

print('\nTuning BordersConsensus...')
print('\tParameter threshold')
tune_borders_consensus_threshold(development_set, param_ranges={'threshold': (0,450)})

print('\nTuning finished!')
END