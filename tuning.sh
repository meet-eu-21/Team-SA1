#!/bin/bash

python - <<END
import platform
print('Init tuning from platform {}'.format(platform.system()))
from src.data import HiCDataset
import src.utils
import src.metrics
import src.tad_algo
from src.tuning import tune_topdom, tune_tadtree

print('Tuning TopDom...')
print('\tParameter window')
tune_topdom(development_set)
print('Tuning TADTree...')
print('\tParameter N')
tune_tadtree(development_set, param_ranges={'N': (100,600)})
print('\tParameter p')
tune_tadtree(development_set, param_ranges={'p': (2,5)})
print('Tuning finished!')
END