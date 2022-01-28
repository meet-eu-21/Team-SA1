<h1 align="center"> Team-SA1 for TAD prediction: BananaTAD <img src="https://img.icons8.com/color/48/000000/banana.png" width="20"/> </h1>

<h2 align="center"><img src="https://img.icons8.com/external-justicon-flat-justicon/64/000000/external-france-countrys-flags-justicon-flat-justicon.png", width="20"/> Meet-EU Team SA1: <img src="https://img.icons8.com/external-justicon-flat-justicon/64/000000/external-france-countrys-flags-justicon-flat-justicon.png", width="20"/> </h2>
<h4 align="center"><img src="https://img.icons8.com/color/48/000000/banana.png" width="20"/> A. Marina, C. Simon, L. Liam, T. Alexis, Z. Yann <img src="https://img.icons8.com/color/48/000000/banana.png" width="20"/></h4> 

## Background
### What is this?
This GitHub page is an ongoing project for the Meet-EU course at Sorbonne University on the creation and improvement of algorithms and methods for TAD detection (explained in more detail below).

The main page (which includes a detailed description, overview by teachers and organisers) for the project can be found [here](https://github.com/hdsu-bioquant/meet-eu-2021).

Input and validation data for the project can be found [here](http://www.lcqb.upmc.fr/meetu/).

A brief intro to TADs by Leopold Carron can be found [here](https://bioinfo-fr.net/quest-ce-quun-tad-topological-associated-domain) (in French).

### What is a TAD?
DNA is commonly referred to as the "building block" of life; it is organised into structures known as chromosomes, which in turn are extremely densely packed molecules that hold all the necessary information for gene expression. The densest parts of DNA are called chromatin, and chromatin that interact with each other to a high degree do so in what is called a TAD: a topologically associated domain, or region.

A TAD can be considered the basic unit of chromosome folding - they are the result of DNA compaction through the action of histones, causing bases that are not in sequential proximity to end up in local proximity.
The primary goal of this project is thus the improvement and analysis of TAD calling tools, algorithms and pipelines from Hi-C sequencing data; we strive to present our results in a clear enough way to hopefully contribute towards this as of yet still ongoing field of research in bioinformatics.

### <img src="https://img.icons8.com/external-filled-outline-geotatah/40/000000/external-document-workmen-compensation-filled-outline-filled-outline-geotatah.png"/>We did it by:
- <img src="https://img.icons8.com/emoji/30/000000/telescope-.png"/> **Reimplementing** various TAD detection algorithms such as TopDom, TAD tree, and Tadbit under python, these algorithms are used to find said TADs by analysing HiC map files. They provide us with the location of the start and end of the TADs.
- <img src="https://img.icons8.com/external-justicon-lineal-color-justicon/30/000000/external-hand-shake-woman-day-justicon-lineal-color-justicon.png"/> **Comparing** the output TADs between them and with other ones such as TADs produced by the Arrow-Head algorithm to create consensus TADs. Consensus TADs are a mean to avoid redundancy in detected TADs. Sometimes, when we use different methods or resolutions, we might detect TADs that are very close in proximity, but that are detected as two different ones. They still might be the same TAD in reality and that's why we need a consensus method to determine if it is the case or not. 
- <img src="https://img.icons8.com/emoji/30/000000/hundred-points.png"/> Using a **scoring system** that aims to make the whole process more precise.
- <img src="https://img.icons8.com/external-vitaliy-gorbachev-lineal-color-vitaly-gorbachev/30/000000/external-ruler-graphic-design-vitaliy-gorbachev-lineal-color-vitaly-gorbachev.png"/> Developing a TAD **length analysis** method to help with the scoring system and maybe try to predict the distribution of TAD lengths later on (This part is very experimental and we hope more qualified researchers can use it as first step for a new method).
- <img src="https://img.icons8.com/color/30/000000/vertical-settings-mixer--v1.png"/> **Tunning** every functions to secure the best performances.

## Sections and folders
**This part needs to be updated**

The respective folders for this GitHub are:
- Data: contains all the HiC raw datas for the different types of cells we studied at different resolutions, plus already detected TADs from the Arrow-Head method. <img src="https://img.icons8.com/color/25/000000/123.png"/>
- pics: contains pictures of our report <img src="https://img.icons8.com/emoji/25/000000/paintbrush-emoji.png"/>
- Results: contains illustrations of our work. <img src="https://img.icons8.com/external-vitaliy-gorbachev-lineal-color-vitaly-gorbachev/25/000000/external-painting-museum-vitaliy-gorbachev-lineal-color-vitaly-gorbachev.png"/>
  - TADs_length_analysis: contains histograms of distribution of TAD lengths, and fitting curves for this metric.
  - ADD FUTUR RESULT FOLDERS HERE
- src: contains the different python scripts that are used for our project.<img src="https://img.icons8.com/color/25/000000/property-script.png"/>


## Code and libraries, dependencies examples (tbd)
For preprocessing and TAD recognition:
```python
import numpy as np
import liam_method as lm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from src.utils import SCN
import pandas as pd
import os, time, logging
from sklearn.preprocessing import scale
from scipy import stats
from scipy.stats import ranksums
from abc import ABC, abstractmethod
```
For TADs length analysis:
```python
import numpy as np
import glob
import urllib3
from lmfit.models import GaussianModel
from math import *
import matplotlib.pyplot as plt
import os
```
`topdom` `arrowhead` `tadtree` `tadtool` `hictool`

## Credit
Icons embedded from [icons8](https://icons8.com/), all rights belonging to their respective owners.
