import os

#Make sure you have run the Simon's preprocess algorithm in the folder you're using
#path="C:\\Users\\Alexis\\Documents\\M2\\Meet-EU\\Alexis (•ㅅ•)"
#os.chdir(path)

from lmfit.models import GaussianModel
from math import *
from src.data import *
from src.tad_algo import *
from src.Analyse_longueur_TADs import *
import glob
import re


#Please enter resolution I'm too lazy to creat an input
topdom=TopDom()
resolution=25000

#Just a function that returns y=f(x) if f() is a Gaussian function
def Gaussian_function(x,A,mu,sigma):
    return (A/(sigma*sqrt(2*pi)))*exp(-((x-mu)**2)/(2*sigma)**2)

#Use this if you want to do a single file
#file_path="C:/Users/Alexis Trang/Documents/Cours_UPMC_M2/MEET-U/Data/HiC/GM12878/25kb_resolution_intrachromosomal/chr1_25kb.npy.txt.npy"

#prod_top_dom_tads is a function that creates topdom TADs analysis results.
#prod_top_dom_tads takes as input the path to the file you want to analyse (file_path=str), the resolution of the HiC map (resolution=int), and the window for the TopDom method(window=5).
#it returns a list with the positions of start and end of the TADs detected (topdom_tads=list of tupples).

def prod_topdom_tads(file_path,resolution=25000,window=5):
    hic_mat = Hicmat(file_path, resolution)
    hic_mat.filter(threshold = 1)
    topdom_tads = topdom.getTADs(hic_mat)
    return topdom_tads

#fitting_curve_list does what fitting_curve_singlefile but takes as input a list of TAD positions (as what prod_topdom_tads returns)

def fitting_curve_list(TAD_list,resolution=25000): #TADs_method should be either "TD" (if your file is to be analysed by TopDom) or "AH" (of analysed by ArrowHead).

    data=extraction_data_TD(TAD_list)
    dico_long=data[2]
    l_long=list(dico_long.keys())
    l_long.sort()
    l_count=[]
    for j in l_long:
        l_count.append(dico_long[j])
    x,y=np.array(l_long),np.array(l_count)           #

    #Do the method with the model = Gaussian
    mod = GaussianModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp,Center,Sigma=out.best_values['amplitude'],out.best_values['center'],out.best_values['sigma']

    #Values of our model
    courbe=[]
    for k in x:
        courbe.append(Gaussian_function(k,Amp,Center,Sigma))

    #plot of the comparison between empirical and predicted datas
    plt.plot(x, y, 'b.-')
    plt.ylabel("Effectives",fontsize=25)
    plt.xlabel("TAD sizes",fontsize=25)
    plt.plot(x,courbe,'g-')
    plt.legend(["Observed TAD sizes","Created model"],fontsize=25)
    plt.show()

    opt=open("parametres_optimaux.txt","a")
    opt.write(str(Amp)+"\t"+str(Center)+"\t"+str(Sigma)+"\n")
    opt.close()

    return Amp,Center,Sigma

from lmfit.models import ExponentialModel


#Just a function that returns y=f(x) if f() is an exponential function
def Exponential_function(x,A,d):
    return A*exp(-x/d)

#Use this if you want to do a single file
#file_path="C:/Users/Alexis Trang/Documents/Cours_UPMC_M2/MEET-U/Data/HiC/GM12878/25kb_resolution_intrachromosomal/chr1_25kb.npy.txt.npy"

#prod_top_dom_tads is a function that creates topdom TADs analysis results.
#prod_top_dom_tads takes as input the path to the file you want to analyse (file_path=str), the resolution of the HiC map (resolution=int), and the window for the TopDom method(window=5).
#it returns a list with the positions of start and end of the TADs detected (topdom_tads=list of tupples).

def prod_topdom_tads(file_path,resolution=25000,window=5):
    hic_mat = Hicmat(file_path, resolution)
    hic_mat.filter(threshold = 1)
    topdom_tads = topdom.getTADs(hic_mat)
    return topdom_tads

#fitting_curve_list does what fitting_curve_singlefile but takes as input a list of TAD positions (as what prod_topdom_tads returns)


def fitting_curve_list_consensus(TAD_list,resolution=25000,Name=None): #TADs_method should be either "TD" (if your file is to be analysed by TopDom) or "AH" (of analysed by ArrowHead).

    data=extraction_data_TD(TAD_list)
    dico_long=data[2]
    l_long=list(dico_long.keys())
    l_long.sort()
    l_count=[]
    for j in l_long:
        l_count.append(dico_long[j])
    x,y=np.array(l_long),np.array(l_count)           #

    #Do the method with the model = Gaussian
    mod = ExponentialModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp,Decay=out.best_values['amplitude'],out.best_values['decay']

    #Values of our model
    courbe=[]
    for k in x:
        courbe.append(Exponential_function(k,Amp,Decay))

    #plot of the comparison between empirical and predicted datas
    plt.plot(x, y, 'b.-')
    #plt.plot(x, out.best_fit, '-', label='best fit')
    plt.ylabel("Effectives",fontsize=10)
    plt.xlabel("TAD sizes",fontsize=10)
    plt.plot(x,courbe,'g-')
    if Name!= None:
        plt.title("Fitting exponentional curve of "+Name)
    plt.legend(["Observed TAD sizes","Created model"],fontsize=10)
    plt.show()

    opt=open("parametres_optimaux.txt","a")
    opt.write(Name+"\tExp\t"+str(Amp)+"\t"+str(Decay))
    opt.close()
    
def fitting_curve_single_file(file_path,TADs_method,resolution=25000,Name_of_cells=None,Chromosome=None): #TADs_method should be either "TD" (if your file is to be analysed by TopDom) or "AH" (of analysed by ArrowHead).


    if TADs_method == "TD":                             #This block extracts
        topdom_tads = prod_topdom_tads(file_path)       #the sizes and number
        if topdom_tads!=[]:
            data_step=extraction_data_TD(topdom_tads)
            dico_long=data_step[2]                                          #of TADs for
            l_path=file_path.split("/")
            lign,res,chr=l_path[-3],re.search(r"[\d]+kb",l_path[-2]).group(),re.search(r"chr[\d]+",l_path[-1]).group()
            name=lign+chr
            Title="Gaussian fitting curve for TopDom TADs, "+res+" resolution,"+" in "+lign+chr+"."

    elif TADs_method=="AH":                           #each sizes detected
        data_step=extraction_data_AH(file_path)
        dico_long=data_step[2]
        name_file=file_path.split("/")[-1]
        name=re.search(r"_[A-Z\d]+_",name_file).group()[1:-1]
        Title="Gaussian fitting curve for ArrowHead TADs, 5kb resolution, in "+name+"."

    elif TADs_method=="TT":
        data_step=extraction_data_TT(file_path,resolution)
        dico_long=data_step[2]
        Title= "Gaussian fitting curve of TADtree, "+str(int(resolution/1000))+"kb resolution, in "+Name_of_cells+"."
        name=Name_of_cells+Chromosome

    l_long=list(dico_long.keys())
    l_long.sort()
    l_count=[]
    for j in l_long:
        l_count.append(dico_long[j])
    x,y=np.array(l_long),np.array(l_count)           #

    #Do the method with the model = Gaussian
    mod = GaussianModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp,Center,Sigma=out.best_values['amplitude'],out.best_values['center'],out.best_values['sigma']

    #Values of our model
    courbe=[]
    for k in x:
        courbe.append(Gaussian_function(k,Amp,Center,Sigma))

    #plot of the comparison between empirical and predicted datas
    plt.plot(x, y, 'b.-')
    plt.ylabel("Effectives",fontsize=25)
    plt.xlabel("TAD sizes",fontsize=25)
    plt.plot(x,courbe,'g-')
    plt.title(Title, fontsize=30)
    plt.legend(["Observed TAD sizes","Created model"],fontsize=25)
    plt.show()

    opt=open("parametres_optimaux.txt","a")
    opt.write(name+"\t"+str(Amp)+"\t"+str(Center)+"\t"+str(Sigma)+"\n")
    opt.close()

    return Amp,Center,Sigma

#fitting_curve_cole_folder does exactly  the same thing as fitting_curve_single_file, but on a whole folder.
def fitting_curve_whole_folder(folder_path,TADs_method,resolution=25000): #TADs_method should be either "TD" (if your file is to be analysed by TopDom) or "AH" (of analysed by ArrowHead).
    count = 0
    dico_long = {}
    if TADs_method == "TD":                             #This block extracts
        preprocess_data(folder_path,resolution)
        for i in glob.glob(folder_path+"/*.npy*"):
            count += 1 #nmbr of files
            topdom_tads = prod_topdom_tads(i)       #the sizes and number
            if topdom_tads!=[]:
                data_step=extraction_data_TD(topdom_tads)
                dico_step=data_step[2]
                for j in list(dico_step.keys()):
                    if j in dico_long.keys():
                        dico_long[j]+=dico_step[j]
                    else:
                        dico_long[j]=dico_step[j]
            l_path=folder_path.split("/")
            lign,res=l_path[-2],re.search(r"[\d]+kb",l_path[-1]).group()
            Title="Gaussian fitting curve for TopDom TADs, "+res+" resolution,"+" in "+lign+"chrall."
                                                      #of TADs for
    elif TADs_method=="AH":                           #each sizes detected

        for i in glob.glob(folder_path+"/*.txt"):
            count+=1 #nmbr of files
            data_step=extraction_data_AH(i)
            dico_step=data_step[2]
            for j in list(dico_step.keys()):
                if j in dico_long.keys():
                    dico_long[j]+=dico_step[j]
                else:
                    dico_long[j]=dico_step[j]
        Title="Gaussian fitting curve for ArrowHead TADs, 5kb resolution."

    elif TADs_method=="TT":
        data_step=extraction_data_TT(file_path,resolution)
        dico_long=data_step[2]
        Title= "Gaussian fitting curve of TADtree, "+str(int(resolution/1000))+"kb resolution, in "+Name_of_cells+"."
        name=Name_of_cells+Chromosome

    elif TADs_method=="CO":
        data_step=

    l_long=list(dico_long.keys())
    l_long.sort()
    l_count=[]
    for j in l_long:
        l_count.append(dico_long[j]/count)
    x,y=np.array(l_long),np.array(l_count)           #

    #Do the method with the model = Gaussian
    mod = GaussianModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp,Center,Sigma=out.best_values['amplitude'],out.best_values['center'],out.best_values['sigma']

    #Values of our model
    courbe=[]
    for k in x:
        courbe.append(Gaussian_function(k,Amp,Center,Sigma))

    #plot of the comparison between empirical and predicted datas
    plt.plot(x, y, 'b.-')
    plt.ylabel("Effectives",fontsize=25)
    plt.xlabel("TAD sizes",fontsize=25)
    plt.plot(x,courbe,'g-')
    plt.title(Title, fontsize=30)
    plt.legend(["Observed TAD sizes","Created model"],fontsize=25)
    plt.show()

    opt=open("parametres_optimaux.txt","a")
    opt.write(str(Amp)+"\t"+str(Center)+"\t"+str(Sigma)+"\n")
    opt.close()

    return Amp,Center,Sigma