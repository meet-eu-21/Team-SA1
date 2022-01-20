import os


#Make sure you have run the Simon's preprocess algorithm in the folder you're using
path="C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)"
os.chdir(path)

#From src provided by Simon import relevant functions
from src.data import *
from src.tad_algo import *
from src.Analyse_longueur_TADs import *

#Please enter resolution I'm too lazy to creat an input
topdom=TopDom()
resolution=25000

#Use this if you want to do a single file
#file_path="C:/Users/Alexis Trang/Documents/Cours_UPMC_M2/MEET-U/Data/HiC/GM12878/25kb_resolution_intrachromosomal/chr1_25kb.npy.txt.npy"

#prod_top_dom_tads is a function that creates topdom TADs analysis results

def prod_topdom_tads(file_path,resolution=25000,window=5):
    hic_mat = Hicmat(file_path, resolution)
    hic_mat.filter(threshold = 1)
    topdom_tads = topdom.getTADs(hic_mat)
    return topdom_tads



