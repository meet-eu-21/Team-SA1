import os

#Make sure you have run the Simon's preprocess algorithm in the folder you're using
#path="C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)"
#os.chdir(path)

from lmfit.models import GaussianModel
from math import *
from src.data import *
from src.tad_algo import *
from src.Analyse_longueur_TADs import *
import glob

#print(glob.glob("C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)\\TAD/*"))

#Just a function that returns y=f(x) if f() is a Gaussian function
def Gaussian_function(x,A,mu,sigma):
    return (A/(sigma*sqrt(2*pi)))*exp(-((x-mu)**2)/(2*sigma)**2)

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


