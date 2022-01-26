#Make sure you have run the Simon's preprocess algorithm in the folder you're using
path="C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)"
os.chdir(path)

#From src provided by Simon import relevant functions
from src.data import *
from src.tad_algo import *
from src.Analyse_longueur_TADs import *
import re
from lmfit.models import GaussianModel
from math import *

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