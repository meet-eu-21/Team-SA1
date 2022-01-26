"""
pwd should be: C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Data\\TAD
GSE63525_HMEC_Arrowhead_domainlist.txt
GSE63525_HUVEC_Arrowhead_domainlist.txt
GSE63525_IMR90_Arrowhead_domainlist.txt
GSE63525_NHEK_Arrowhead_domainlist.txt
"""
import os
#Make sure you have run the Simon's preprocess algorithm in the folder you're using
#path="C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)"
#os.chdir(path)
import glob

from src.data import *
from src.tad_algo import *
import matplotlib.pyplot as plt
import numpy as np

def dictionnary_length(name_file):
    dico_long={}
    file=open(name_file,'r')
    line=file.readline()
    line=file.readline().split()
    while line!=[]:
        start,end=int(line[1]),int(line[2])
        if end-start in dico_long.keys():
            dico_long[end-start]+=1
        else:
            dico_long[end-start]=1
        line=file.readline().split()
    return dico_long



def extraction_data_AH(name_file):          #####DATA EXTRACTION FOR ARROWHEAD
    dico_long=dictionnary_length(name_file)
    l_long=list(dico_long.keys())
    l_long.sort()
    l_count=[]
    for j in l_long:
        l_count.append(dico_long[j])
    esp,var,std=np.mean(l_long),np.nanvar(l_long),np.nanstd(l_long)
    return l_long,l_count,dico_long,esp,var,std





def extraction_data_TD(topdom_tads):         ########## PLOT FOR TOP DOM
    l_long=[]
    l_count=[]
    dico_long={}
    for i in topdom_tads:
        if i[1]-i[0] in dico_long.keys():
            dico_long[i[1]-i[0]]+=1
        else:
            dico_long[i[1]-i[0]]=1
    l_long=list(dico_long.keys())
    l_long.sort()
    for j in l_long:
        l_count.append(dico_long[j])
    esp,var,std=np.mean(l_long),np.nanvar(l_long),np.nanstd(l_long)
    return l_long,l_count,dico_long,esp,var,std


def extraction_data_TT(name_file, res):
    l_long=[]
    l_count=[]
    dico_long={}

    file=np.loadtxt(name_file,dtype='str')
    file=np.array(file)
    tadtree_tads_lengths=file[1:,2].astype(int)*res-file[1:,1].astype(int)*res

    for i in tadtree_tads_lengths:
        if i in dico_long.keys():
            dico_long[i]+=1
        else:
            dico_long[i]=1
    l_long=list(dico_long.keys())
    l_long.sort()
    for j in l_long:
        l_count.append(dico_long[j])
    esp,var,std=np.mean(l_long),np.nanvar(l_long),np.nanstd(l_long)
    return l_long,l_count,dico_long,esp,var,std


def Histo_sizes_TADs(l_long,l_count):                ############# PLOT HIST
    plt.bar(range(len(l_count)),l_count,tick_label=l_long)
    plt.show()
    return None




#Please enter resolution I'm too lazy to creat an input
topdom=TopDom()

#prod_top_dom_tads is a function that creates topdom TADs analysis results
#it takes as input a .npy file

def prod_topdom_tads(file_path,resolution=25000,window=5):      ######PRODUCTION OF TADS FROM TOPDOM ALGORITHM
    hic_mat = Hicmat(file_path, resolution)
    hic_mat.filter(threshold = 1)
    topdom_tads = topdom.getTADs(hic_mat)
    return topdom_tads

def prod_topdom_tads_whole_folder(folder_path, resolution=25000,window=5):
    l_topdom_tads=[]
    for i in glob.glob(folder_path+"/*.npy*"):
        l_topdom_tads+=prod_topdom_tads(i,resolution,window)
    return l_topdom_tads





































