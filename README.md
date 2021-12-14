<h1 align="center"> Team-SA1 for TAD prediction <img src="https://img.icons8.com/color/48/000000/predcit.png"/> (algorithms, tools and methods) </h1>

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

## Ongoing work
- Topdom [Publication](https://pubmed.ncbi.nlm.nih.gov/26704975/) and [code](https://github.com/HenrikBengtsson/TopDom), originally coded in R and recoded in Python by the team. <img src="https://img.icons8.com/external-becris-flat-becris/64/000000/external-r-data-science-becris-flat-becris.png" width="20"/> <img src="https://img.icons8.com/color/48/000000/arrow--v2.png" width="20"/> <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/>
- TADTree [Publication](https://academic.oup.com/bioinformatics/article/32/11/1601/1742546?login=true) and [code](https://github.com/raphael-group/TADtree), coded in Python and adapted for our use. <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/> 
- Consensus between the above pipelines; based on multiple criterion such as CTCF sites, TAD overlap between pipelines, TAD size categorization, etc. The consensus is the most important part of the project and we would ideally incorporate more than just the two above pipelines. <img src="https://img.icons8.com/emoji/48/000000/folded-hands-emoji.png" width="20"/> <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/>
- TAD size classification; by studying the size distributions of TADs identified by each pipeline, we hope to be able to classify TADs into main TADs, sub-TADs, etc., and give confidence scores of some sort. <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/>
- ... and more

## Upcoming work
- [TADTool](https://github.com/vaquerizaslab/tadtool): a seemingly comprehensive set of tools that could be useful for improving TAD quantification by identifying meaningful parameters. <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/>
- [HiCtool](https://github.com/Zhong-Lab-UCSD/HiCtool): another promising pipeline to identify TADs from raw Hi-C data. <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/>
- [TADBit](https://github.com/3DGenomes/TADbit): a python library that was also included in Rao et al.'s assessement of TAD calling tools. <img src="https://img.icons8.com/color/48/000000/python--v1.png" width="20"/>
- CTCF sites integration to consensus analysis. <img src="https://img.icons8.com/external-icongeek26-linear-colour-icongeek26/64/000000/external-biology-science-and-technology-icongeek26-linear-colour-icongeek26.png" width="20"/>
- [TADCompare](https://bioconductor.org/packages/devel/bioc/vignettes/TADCompare/inst/doc/TADCompare.html) and [input format](https://bioconductor.org/packages/devel/bioc/vignettes/TADCompare/inst/doc/Input_Data.html): a very recent and promising package in R that allows identification of differential TAD boundaries b/ two datasets and has potential for consensus calling. <img src="https://img.icons8.com/external-becris-flat-becris/64/000000/external-r-data-science-becris-flat-becris.png" width="20"/>
- ... and more

## Sections and folders
**This part needs to be updated**

The respective folders for this GitHub are:
- Arrowhead
- TADTree
- Topdom
- tutorial_notebook
- ... and more

## Code and libraries, dependencies examples (tbd)
```python
import numpy as np
import liam_method as lm
```
`topdom` `arrowhead` `tadtree` `tadtool` `hictool`

## Credit
Icons embedded from [icons8](https://icons8.com/), all rights belonging to their respective owners.
