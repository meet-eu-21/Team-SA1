"""
pwd should be: C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Donnés\\TAD
GSE63525_HMEC_Arrowhead_domainlist.txt
GSE63525_HUVEC_Arrowhead_domainlist.txt
GSE63525_IMR90_Arrowhead_domainlist.txt
GSE63525_NHEK_Arrowhead_domainlist.txt
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def dictionnary_length(name_file):
    dic={}
    file=open(name_file,'r')
    line=file.readline()
    line=file.readline().split()
    while line!=[]:
        #print(line)
        K,start,end=line[0],int(line[1]),int(line[2])
        if K in dic.keys():
            dic[K].append([(start,end),end-start])
        else:
            dic[K]=[[(start,end),end-start]]
        line=file.readline().split()
    return dic



def plot_moustache(name_file):
    dic=dictionnary_length(name_file)
    l_long=[]
    for i in dic.keys():
        for j in dic[i]:
            l_long.append(j[1])
    plt.boxplot(l_long)
    plt.show()
    return l_long

def plot_curve(name_file):
    dic=dictionnary_length(name_file)
    l_long=[]
    for i in dic.keys():
        for j in dic[i]:
            l_long.append(j[1])
    l_long.sort()
    plt.plot(range(len(l_long)),l_long)
    plt.show()
    return l_long

def plot_bar(name_file):
    dic=dictionnary_length(name_file)
    l_long=[]
    l_count=[]
    for i in dic.keys():
        for j in dic[i]:
            l_long.append(j[1])
    l_long.sort()
    sizes=list(set(l_long))
    sizes.sort()
    print(sizes)
    for k in sizes:
        l_count.append(l_long.count(k))
    #print(sizes)
    plt.bar(range(len(l_count)),l_count,tick_label=sizes)
    plt.show()
    esp,var,std=np.mean(l_long),np.nanvar(l_long),np.nanstd(l_long)
    return l_long,l_count,esp,var,std



os.chdir("C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Donnés\\TAD")