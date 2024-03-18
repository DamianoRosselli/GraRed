
import astropy.cosmology as asco
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import funcs
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import math
import operator
import os
import numpy as np
from scipy.integrate import quad,dblquad,nquad
from scipy.optimize import curve_fit
from astropy import constants as const
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline as spline


#### Functions to compute correction for surface brightness

def mod_dist(z,cosmo):
    return 5 * np.log10((1 + z) * cosmo.comoving_distance(z).value) + 25

def lum_func(x,cosmo):
   
    phi=0.0093 * cosmo.h**3 
    M=-20.71 + 5*np.log10(cosmo.h) 
    a=-1.26
    expo=np.exp(-10**(0.4*(M-x)))
    power=10**(0.4*(M-x)*(a+1))
    return 0.4*np.log(10)*phi*power*expo

def cumul_lum(mag,cosmo):
    return quad(lum_func,-100,mag,args=(cosmo))[0]

def vec_cum_lum():
    return np.vectorize(cumul_lum)


def maglim(z,cosmo,survey_lim_mag):
    return survey_lim_mag-mod_dist(z,cosmo)+(2.5*np.log10(1.1))
   
   
def den_SB(z,cosmo,survey_lim_mag):
    return (z**2) * cumul_lum(maglim(z,cosmo,survey_lim_mag),cosmo)

def num_SB(z,cosmo,survey_lim_mag):
    return (z**2) *  lum_func(maglim(z,cosmo,survey_lim_mag),cosmo)


def compute_deltaz(z1,z2,cosmo,survey_lim_mag):
    nume=quad(num_SB,z1,z2,args=(cosmo,survey_lim_mag))[0]
    deno=quad(den_SB,z1,z2,args=(cosmo,survey_lim_mag))[0]
    return nume/deno


##### Function for compute model in GR

def g_c5(c):
    return 1./(np.log(1+c)-(c/(1+c)))



def NFW_2d(x,c,R) :#funzione da integrare per avere profilo NFW 2D
    return 1./(((1+c*x)**2)*np.sqrt(x**2-R**2))

def potential(x,c,R): #funzione da integrare per avere il potenziale gravitazionale 
    return (-c + (np.log(1+c*x)/x))*NFW_2d(x,c,R)

def NUM(x,mass_func):
    return x*np.power(10,mass_func(np.log10(x)))


def DEN(x,mass_func):
    return np.power(x,1./3.)*np.power(10,mass_func(np.log10(x)))

def potential_int(c, R):

    return quad(potential, R, np.inf, args=(c, R))[0]

def NFW_int(c, R):

    return quad(NFW_2d, R, np.inf, args=(c, R))[0]

def TD(x,c,R):
    
    return ((-c/(x*(1+c*x)))+(np.log(1+c*x)/(x**2)))*(1./(x*((1+c*x)**2)))*(np.sqrt(x**2-R**2))
    
def TD_int(c, R):
    
    return quad(TD, R, np.inf, args=(c, R))[0]


vec_td_int = np.vectorize(TD_int)
vec_pot_int = np.vectorize(potential_int)
vec_nfw_int = np.vectorize(NFW_int)




def interp_mass_func(m500,islog=False,nbin=30):
    if islog:
        mass = 10**(m500)
    else:
        mass = m500


    bins=np.logspace(np.min(np.log10(mass)),np.max(np.log10(mass)),nbin)
    hist,_=np.histogram(mass,bins)


    mean_bin=np.empty(len(hist),dtype=float)
    hist_dm=np.empty(len(hist),dtype=float)
    error=np.empty(len(hist),dtype=float)

    for i in range(len(hist)):
        hist_dm[i]=hist[i]/((bins[i+1]-bins[i]))
        mean_bin[i]=(bins[i]+bins[i+1])*0.5
        error[i]=np.sqrt(histm[i])/((bins[i+1]-bins[i]))

    error=error/(hist_dm*np.log(10))
    logbin=np.log10(mean_bin)
    loghist=np.log10(hist_dm)

    idx = np.isfinite(loghist)
    loghist=loghist[idx]
    logbin=logbin[idx]
    error=error[idx]

    ff = spline(logbin,loghist,error)

    return spline



def compute_gra_red_gr_model(z,r500,m500,c500,mass_func,cosmo=asco.Planck18,islog=False,survey_lim_mag=None,compute_fr=False):
   
    if islog:
        mass = 10**(m500)
    else:
        mass = m500

    if survey_lim_mag is not None:
        deltaz=compute_deltaz(np.min(z),np.max(z),cosmo,survey_lim_mag)
    else:
        deltaz=0.


    rho_cn=cosmo.critical_density(z).to('kg/m3')


    c500_median=np.median(c500)
    rho_median=np.median(rho_cn)
    M500_min=np.min(mass)
    M500_max=np.max(mass)
    rho_msun=cosmo.critical_density(z).to('Msun/Mpc3')
    rho_m_msun=np.median(rho_msun)
    rho_std_msun=np.std(rho_msun)

    rr=np.geomspace(0.0001,4.0,50)

    const_fac=(4*np.pi/3)*np.power(3/(4*np.pi),2/3)*const.G*np.power(const.M_sun,2./3.)/(const.c)


    num_r=vec_pot_int(c500_median,rr)
    num_m=quad(NUM,M500_min,M500_max,args=(mass_func))[0]
    num=num_r*num_m

    den_r=vec_nfw_int(c500_median,rr)
    den_m=quad(DEN,M500_min,M500_max,args=(mass_func))[0]
    den=den_m*den_r

    redshift=(const_fac*(num/den)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    redshift_gr=redshift.value

    #model TD
    num_r=vec_td_int(c500_median,rr)
    num_m=quad(NUM,M500_min,M500_max,args=(mass_func))[0]
    num_td=num_r*num_m

    
    redshift_td=((3./2.)*const_fac*(num_td/den)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    redshift_gr_td=redshift_td.value

    finale=redshift_gr+(redshift_gr_td*(4-(10*deltaz))/3)

    if compute_fr:
        FR=(redshift_gr*4./3.)+((2-(5*deltaz))*8*redshift_gr_td/9.)
    else:
        FR = 0.

    return rr, redshift_gr,redshift_gr_td, finale, FR


### FUnctions to compute model in  DGP 

def massnfw(x,c,rho):
    a=4./3.*rho*np.pi*500
    b= (a*(c**2 )* g_c5(c) *( x**2)) / (x*(1+c*x)**2)
    return b


def massnfw_int(c,R,rho):
    return quad(massnfw, 0., R, args=(c,rho))[0]



def g_dgp(x,c,m,R,rho):
    mm=massnfw_int(c,R,rho)
    c=m/mm
    xx= np.power((1./160.),1./3.)*x*(np.power(c,1./3.))
    a=  (xx**3) * (np.sqrt(1+(1./(xx**3)))-1)
    b= 1-(a*0.5797101449275)
    return b

    

def potential_dgp(x,c,m,R,rho):
    return potential(x,c,R)*g_dgp(x,c,m,R,rho)


def f(y,x,c,R,rho,mass_func):
    return  NUM(x,mass_func)*potential_dgp(y,c,x,R,rho)


def numdgp_int(c,R,rho,mass_func,m_min,m_max):  
    return nquad(f,[[R,np.inf],[m_min,m_max]], args=(c,R,rho,mass_func))[0]


def ftd(y,x,c,R,rho,mass_func):
    return NUM(x,mass_func)*TD(y,c,R)*g_dgp(y,c,x,R,rho)

def tddgp_int(c,R,rho,mass_func,m_min,m_max):  
    return nquad(ftd,[[R,np.inf],[m_min,m_max]], args=(c,R,rho,mass_func))[0]

vec_numdgp=np.vectorize(numdgp_int)
vec_tddgp=np.vectorize(tddgp_int)



def compute_gra_red_DGP_model(z,r500,m500,c500,mass_func,cosmo=asco.Planck18,islog=False,survey_lim_mag=None):
   
    if islog:
        mass = 10**(m500)
    else:
        mass = m500

    if survey_lim_mag is not None:
        deltaz=compute_deltaz(np.min(z),np.max(z),cosmo,survey_lim_mag)
    else:
        deltaz=0.


    rho_cn=cosmo.critical_density(z).to('kg/m3')


    c500_median=np.median(c500)
    rho_median=np.median(rho_cn)
    M500_min=np.min(mass)
    M500_max=np.max(mass)
    rho_msun=cosmo.critical_density(z).to('Msun/Mpc3')
    rho_m_msun=np.median(rho_msun)
    rho_std_msun=np.std(rho_msun)

    rr=np.geomspace(0.0001,4.0,50)


    const_fac_dgp=0.96*(4*np.pi/3)*np.power(3/(4*np.pi),2/3)*const.G*np.power(const.M_sun,2./3.)/(const.c)


    num=vec_numdgp(c500_median,rr,rho_m_msun.value,mass_func,M500_min,M500_max)
    
    den_r=vec_nfw_int(c500_median,rr)
    den_m=quad(DEN,M500_min,M500_max,args=(mass_func))[0]
   
    den=den_m*den_r
    nude=num/den
   
    
    redshift=(const_fac_dgp*(nude)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')

    #model TD
    num_td=vec_tddgp(c500_median,rr,rho_m_msun.value,mass_func,M500_min,M500_max)
    nude_td=num_td/den
    

    redshift_td=((3./2.)*const_fac_dgp*(nude_td)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    finale=redshift+((2-(5*deltaz))*2*redshift_td/3.)
    return rr,redshift,redshift_td,finale
    

    







      



