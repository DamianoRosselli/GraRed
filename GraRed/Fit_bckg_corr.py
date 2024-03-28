import numpy as np
import matplotlib.pyplot as plt
from . import utils as ut
import itertools
import iminuit
from iminuit import cost, Minuit

def create_histo(r,v,rbin_size,vbin_size):

        #define number of bins for distances and velocities to use within the histogram
        nbinsr=round((np.max(r)-np.min(r))/(rbin_size))
        nbinsv=round((np.max(v)-np.min(v))/(vbin_size))
        nbins=[nbinsr,nbinsv]

        #collect all the occurences in a histogram-variable
        hist,xedges,yedges=np.histogram2d(r,v,bins=nbins)

        #find the central value of each bin for both distances and velocities
        mean_binsr=np.empty(len(xedges)-1,dtype=float)
        mean_binsv=np.empty(len(yedges)-1,dtype=float)

        for i in range(len(xedges)-1):
                mean_binsr[i]=(xedges[i]+xedges[i+1])*0.5

        for i in range(len(yedges)-1):
                mean_binsv[i]=(yedges[i]+yedges[i+1])*0.5

        return hist,mean_binsr,mean_binsv


#define the function with which the fit was performed 
def fit_func(rr,vv,p,func_name):

	#linear fit 2d	
	if func_name=="Lin_2":
		a,b,c,d,e,f=p
		g = a*rr**2+b*vv**2+c*rr*vv+d*vv+e*rr+f
		
		return g
	
	else:
		raise ValueError ("sorry, at the moment we want just a linear fit 2d Lin_2")


def fit_histo(h,rbin,vbin,v_cut,func_name):
	
	#select the range of backround galaxies velocities to perform the fit5
	idx=np.where(abs(vbin)>v_cut)
	vbin1=vbin[idx]
	#select the part of the histogram where to perform the fit
	hist1=h[:,idx[0]]    
	#create a 2Dgrid of number_of_distances_bins x number_of_velocities_bins
	rx,vy=np.meshgrid(rbin,vbin)
	rx1,vy1=np.meshgrid(rbin,vbin1)

	#define chi squared and fit the histogram with a polynomial function of grade 2
	def chi2(a,b,c,d,e,f):
		p=np.array([a,b,c,d,e,f])
		g = fit_func(rx1,vy1,p,func_name)
		chi2 =((hist1.T - g)**2).sum()
			
		return chi2

                
    #minimize chi squared previously defined
	chimin = iminuit.Minuit(chi2, a=1,b=1,c=1,d=1,e=1,f=1)
	chimin.fixed = [False,False,False,False,False,False]
	chimin.errordef = iminuit.Minuit.LEAST_SQUARES
	chimin.migrad()
	chimin.hesse()
	chimin.minos()
	
	par=[chimin.params["a"].value,chimin.params["b"].value,chimin.params["c"].value,chimin.params["d"].value,chimin.params["e"].value,chimin.params["f"].value]
	
	return fit_func(rx,vy,par,func_name),par


def hist_bckg_rem(histo,fit):
	
	#initialize an empty histogram
	new_hist=np.empty((len(histo.T[:,0]),len(histo.T[0,:])),dtype=float)
	
	#fill the histogram with the difference between the original one and the fit function
	for i in range(len(fit[:,0])):
		for j in range(len(fit[0,:])):
			if (fit)[i,j]>=0:
				new_hist[i,j]=histo.T[i,j]-fit[i,j]
				
			else:
				new_hist[i,j]=histo.T[i,j]

	#ensure that occurences are not negative 
	idx_nh=np.where(new_hist<0)
	new_hist[idx_nh]=0
	
	return new_hist



def histo_cut_norm(n_slices,h,f,rbin,vbin):
	
	new_histo_n=np.empty([n_slices,len(vbin)],dtype=float)

	#cut the histogram without background in slices according to the distance range
	for i in range(n_slices):
		new_histo_n[i,]=np.sum(hist_bckg_rem(h,f)[:,round(i*len(rbin)/n_slices):round((i+1)*len(rbin)/n_slices)],axis=1)  
		
		#normalize the histogram as it was a PDF, taking into account velocity bins
		new_histo_n[i,]=new_histo_n[i,]/(np.sum(new_histo_n[i,])*((np.max(vbin)-np.min(vbin))/len(vbin)))

	return new_histo_n


#compute the rms on the residual of the histogram
def std_res(hh,vbin,v_cut):

	#select the range where the fit has been performed
	idx=np.where(abs(vbin)>v_cut)
	hh=hh[idx[0],:]

	return np.std(hh)


def err_norm(n_slices,h,f,rbin,vbin,v_cut):
	
	new_histo_n=np.empty([n_slices,len(vbin)],dtype=float)
	err=np.empty([n_slices,len(vbin)],dtype=float)

	#cut the histogram without background in slices according to the distance range
	for i in range(n_slices):
		new_histo_n[i,]=np.sum(hist_bckg_rem(h,f)[:,round(i*len(rbin)/n_slices):round((i+1)*len(rbin)/n_slices)],axis=1) 
		#write the error as the squared sum of rms (calculated on the residuals) and the poissonian noise of the counts
		err[i,]=np.sqrt(std_res(h.T,vbin,v_cut)**2+new_histo_n[i,])
		err[i,]=err[i,]/(np.sum(new_histo_n[i,])*((np.max(vbin)-np.min(vbin))/len(vbin)))
	
	return err
