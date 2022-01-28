<h1 align="center"> Team-SA1 for TAD prediction: Banana-TAD <img src="https://img.icons8.com/color/48/000000/banana.png" width="20"/> </h1>

<h2 align="center"><img src="https://img.icons8.com/external-justicon-flat-justicon/64/000000/external-france-countrys-flags-justicon-flat-justicon.png", width="20"/> Meet-EU Team SA1: <img src="https://img.icons8.com/external-justicon-flat-justicon/64/000000/external-france-countrys-flags-justicon-flat-justicon.png", width="20"/> </h2>
<h4 align="center"> A. Marina, C. Simon, L. Liam, T. Alexis, Z. Yann

## Background
### What is this?
This GitHub page is a project for the Meet-EU course at Sorbonne University on the creation and improvement of algorithms and methods for TAD detection (explained in more detail below).

The main page (which includes a detailed description, overview by teachers and organisers) for the project can be found [here](https://github.com/hdsu-bioquant/meet-eu-2021).

Input and validation data for the project can be found [here](http://www.lcqb.upmc.fr/meetu/).

A brief intro to TADs by Leopold Carron can be found [here](https://bioinfo-fr.net/quest-ce-quun-tad-topological-associated-domain) (in French).


### What is a TAD?
DNA is a greatly complex and important molecule that is often referred to as the building block of life. The entire human genome is able to fit within a single nucleus, and this is in part thanks to how efficiently DNA is able to pack and compress itself while still retaining high levels of necessary interactions for gene expression. One such unit necessary for this function is known as a Topologically Associating Domain (TAD) – a genomic region capable of high self-interaction through the presence of chromatin and loop formation. TADs are a discovery made within the last decade after the advent of chromosome conformation techniques such as Hi-C sequencing.

The recency in TADs’ discoveries has come with many questions and challenges associated to it. TADs are important because their presence is often associated to the regulation of gene expression, and disruption in TADs has been linked to oncogenic onset, for example. However, TADs are not trivial to detect. There have been many algorithms and methods such as Topdom or Arrowhead, to cite the two most known, but because of a general lack of consensus on what exactly a TAD is, this field still faces many challenges in the vital aspect of “calling” (or identifying) a TAD from raw sequencing data.

Our method, Banana-TAD, seeks to help solve this challenge by the re-implementation of four existing algorithms: Topdom, TADTree, OnTAD as well as TADBit, and scoring them based on a set of data provided to us as ground truth (as lists of TADs called by Arrowhead on 5kb data). After scoring them, we then obtained a list of consensus TADs, which we believe to be a strong middle ground obtained from the combination of multiple recognized TAD calling methods.

## Methods
### <img src="https://img.icons8.com/external-filled-outline-geotatah/40/000000/external-document-workmen-compensation-filled-outline-filled-outline-geotatah.png"/>Project steps:
- <img src="https://img.icons8.com/emoji/30/000000/telescope-.png"/> **Reimplementing** various TAD detection algorithms such as TopDom, TAD tree, OnTAD and TADBit under Python, these algorithms are used to find said TADs by analysing HiC map files. They provide us with the location of the start and end of the TADs.
- <img src="https://img.icons8.com/external-justicon-lineal-color-justicon/30/000000/external-hand-shake-woman-day-justicon-lineal-color-justicon.png"/> **Comparing** the output TADs between them and with other ones such as TADs produced by the Arrow-Head algorithm to create consensus TADs. Consensus TADs are a mean to avoid redundancy in detected TADs. Sometimes, when we use different methods or resolutions, we might detect TADs that are very close in proximity, but that are detected as two different ones. They still might be the same TAD in reality and that's why we need a consensus method to determine if it is the case or not. 
- <img src="https://img.icons8.com/emoji/30/000000/hundred-points.png"/> Using a **scoring system** that aims to make the whole process more precise.
- <img src="https://img.icons8.com/external-vitaliy-gorbachev-lineal-color-vitaly-gorbachev/30/000000/external-ruler-graphic-design-vitaliy-gorbachev-lineal-color-vitaly-gorbachev.png"/> Developing a TAD **length analysis** method to help with the scoring system and maybe try to predict the distribution of TAD lengths later on (This part is very experimental and we hope more qualified researchers can use it as first step for a new method).
- <img src="https://img.icons8.com/color/30/000000/vertical-settings-mixer--v1.png"/> **Tuning** every functions to secure the best performances based on defined metrics.

## Sections and folders
**This part needs to be updated**

The respective folders for this GitHub are:
- Data: contains all the HiC raw datas for the different types of cells we studied at different resolutions, plus already detected TADs from the Arrow-Head method. <img src="https://img.icons8.com/color/25/000000/123.png"/>
- pics: contains pictures of our report <img src="https://img.icons8.com/emoji/25/000000/paintbrush-emoji.png"/>
- Results: contains illustrations of our work. <img src="https://img.icons8.com/external-vitaliy-gorbachev-lineal-color-vitaly-gorbachev/25/000000/external-painting-museum-vitaliy-gorbachev-lineal-color-vitaly-gorbachev.png"/>
  - TADs_length_analysis: contains histograms of distribution of TAD lengths, and fitting curves for this metric.
  - ADD FUTUR RESULT FOLDERS HERE
- src: contains the different python scripts that are used for our project.<img src="https://img.icons8.com/color/25/000000/property-script.png"/>


## Command line to run Banana-TAD on a single 100kb file
The following example should be run after cloning the repo locally. This command saves the list of all consensus TADs of the chromosome at 100kb resolution as tuples (start, end).
```bash
python BananaTAD.py --folder data\example --file chr1_100kb.RAWobserved --cell_type GM12878 --resolution 100000 --chrom 1
```
`arrowhead` `topdom` `tadtree` `ontad` `tadbit`

## Credit
Icons embedded from [icons8](https://icons8.com/), all rights belonging to their respective owners.
