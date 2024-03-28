import numpy as np
from astropy import constants as cst
import scipy
from scipy.special import erf
from scipy.optimize import fsolve,root
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.spatial.distance import cdist
light_vel=cst.c.to('km/s').value

def g(c):
    return np.log(1+c)-(c/(1+c))

def c500_interp(x,m5,m2,c2):
    return (m2/m5)*g(x)-g(c2)

def vlos(zcen,zgal):
    vl = np.log(1+zgal) - np.log(1+zcen)
    return vl*light_vel

def vlos_center(z1,z2):
    # z1 cluster center redshift, z2 galaxy redshift
    vl = cdist(z1.reshape(-1,1), z2.reshape(-1,1), metric=lambda x, y: np.log(1+y)-np.log(1+x))
    return vl*light_vel

def vlos_err(z1,z2,dz1,dz2):
    err1=dz1/(1+z1)
    err2=dz2/(1+z2)
    vl_err = cdist(err1.reshape(-1,1), err2.reshape(-1,1), metric=lambda x, y: np.sqrt(x**2 + y**2))
    return vl_err*light_vel

def comoving_distance(z,cosmology):
    return cosmology.comoving_distance(z)

def comoving_transverse_distance(z,cosmology):
    return cosmology.comoving_transverse_distance(z)

def dist_skyobj(u,v,dm1=None,r5=None,Mpc=True,arcsec=False,normalized=True,radians=False):
    	# u = angular coordinates of cluster center in radians
    	#v = angular coordinates of galaxies in radians
    	# dm1 = comoving transverse distance in Mpc

    if not radians:
    	racl= np.radians(racl)
    	deccl= np.radians(deccl)
    	ragal= np.radians(ragal)
    	decgal= np.radians(decgal)

    
    a = cdist(u[:,1].reshape(-1,1), v[:,1].reshape(-1,1), metric=lambda x, y: np.sin((y-x)*0.5)**2)

    b= cdist(u[:,0].reshape(-1,1), v[:,0].reshape(-1,1), metric=lambda x, y: np.cos(x)*np.cos(y)*(np.sin((y-x)*0.5))**2)
    
    theta=2*np.arcsin(np.sqrt(a+b))
    
    if Mpc and arcsec:
        raise ValueError('choose the distance in Mpc or arcesc?')

    elif Mpc and not arcsec:
        if dm1 is None:
            raise ValueError('give transverse comoving ditance in Mpc')
        else:
            d=(theta.T*np.array(dm1)).T

    elif not Mpc and arcesc:
        d= theta * 206264.806
        
    if not normalized:
        return d
    elif normalized:
        if r5 is None:
            raise ValueError('give the radius to normalize the distances')
        else:
            return (d.T/r5).T

def dist_sky(racl,deccl,ragal,decgal,dm1=None,r5=None,Mpc=True,arcsec=False,normalized=True,radians=False):
    	  
   
    if not radians:
    	racl= np.radians(racl)
    	deccl= np.radians(deccl)
    	ragal= np.radians(ragal)
    	decgal= np.radians(decgal)
   
    a = np.sin((0.5*(decgal-deccl))**2)
    b = np.cos(racl)*np.cos(ragal)*(np.sin((ragal-racl)*0.5))**2
    theta=2*np.arcsin(np.sqrt(a+b))
    
    if Mpc and arcsec:
        raise ValueError('choose the distance in Mpc or arcesc?')

    elif Mpc and not arcsec:
        if dm1 is None:
            raise ValueError('give transverse comoving ditance in Mpc')
        else:
            d=theta*dm1

    elif not Mpc and arcesc:
        d= theta * 206264.806
        
    if not normalized:
        return d
    elif normalized:
        if r5 is None:
            raise ValueError('give the radius to normalize the distances')
        else:
            return d/r5

def P2mitchell(fr0,redshift,cosmology):
    fracomega=cosmology.Ode0/cosmology.Om0
    c1c2=-1*((3*(1+4*fracomega))**2)*fr0
    mR=1./(3*(((1+redshift)**3)+4*fracomega))
    fr=-1*c1c2*(mR**2)
    p2=1.503*np.log10((np.abs(fr)/(1+redshift)))+21.64
    return p2


def convert_concentration(mass,redshift,fr0,cosmology):
    lamb=0.55
    epss=-0.27
    oms=1.7
    alpha=-6.5
    gamma=-0.07
    omt=1.3
    epst=0.1
    p2=P2mitchell(fr0,redshift,cosmology)
    x=np.log10(mass/(10**p2))
    x1=(x-epss)/oms
    a= 1 - np.tanh(omt*(x+epst))
    b=erf(alpha*x1/np.sqrt(2))
    c=gamma + (lamb/oms*gauss_func(x1)*(1+b))
    return 0.5*(a*c)


def gauss_func(x):
    return np.exp(-(x**2)/2)/np.sqrt(2*np.pi)
    
   
def convert_mass_GRtofR(mass,redshift,fr0,cosmology):
    """Covert mass in GR to ones in f(R) following empirical relation from Mitchell et al.
	---- Parameters -----
	mgr M500 cluster mass in solar masses
     	fr0 constant factor of fr theory
     	---Returns---
     	cluster mass in solar masses"""
    p2=P2mitchell(fr0,redshift,cosmology)
    fac=7/6 - (np.tanh(2.21*(np.log10(mass)-p2))/6)
    mfr=(mass)*np.sqrt(fac)
    return mfr

def halo_radius(mass,redshift,delta,cosmology,is_crit):
    
    if is_crit:
        rhoc=cosmology.critical_density(redshift).value
    else:
        rhoc=cosmology.critical_density(redshift).value/cosmology.Odm(redshift)

    rhoc=rhoc*6.77*1.e41
    c=4*np.pi*delta*rhoc
    return np.power((mass*3)/c,1./3.)
    

def compute_c500(z,m500,seed=1234, convert=None):

    logM_min=np.log10(np.min(m500))
    logM_max=np.log10(np.max(m500))
    sample_m=np.logspace(logM_min-1,logM_max+1,500000)
    rand_gen = np.random.default_rng(seed)

    msample=rand_gen.choice(sample_m,size=len(z),replace=False)
    
    del sample_m
    c200=Duffy_NFW_conc(msample,z,'200')
    if convert is not None:
        c200 *= (10**convert)
    
    m500c = {i: np.asarray([Mass_Delta(m,200,500,c,True)[0] for m,c in zip(msample,c200)]) for i,zz in enumerate(z) }
    
    ff = [spline(np.sort(m500c[k]),msample[np.argsort(m500c[k])]) for k in m500c.keys()]
    m200c= {i: f(m500[i]) for i,f in enumerate(ff)}
    c200c = np.asarray([Duffy_NFW_conc(m200c[k],z[k],'200') for k in m500c.keys()]) 

    if convert is not None:
        c200c*=(10**convert)

    c500=[fsolve(c500_interp,1,args=(m500[k],m200c[k],c200c[k]),maxfev=2000)[0] for k in m200c.keys()]
    
    return c500

def Mass_Delta(mass,delta_in,delta_out,conc,is_input):
    ratio=fsolve(mass_interp, 1.,args=(delta_in,delta_out,conc,is_input),maxfev=2000)
    return mass*((delta_out/delta_in)/np.power(ratio, 3))

def mass_interp(xx,delta_in,delta_out,conc,is_input):
    if is_input:
        c_in=conc
        c_out=conc/xx
    else:
        c_in=conc*xx
        c_out=conc
        
    AA = np.log(1.+c_out)-(c_out/(1.+c_out))
    BB = np.log(1.+c_in)-(c_in/(1.+c_in))
    return np.abs(((delta_in/delta_out)*AA*np.power(xx, 3))-BB)

    

    
def Duffy_NFW_conc(mass,redshift,halo_def):
    #from duffy et al 2008 table1 NFW full sample 0<z<2
    if halo_def=='200':
        AA = 5.71
        BB = -0.084
        CC = -0.47

    elif halo_def=='vir':
        AA = 7.85
        BB = -0.081
        CC = -0.71
    elif halo_def=='mean':
        AA = 10.14
        BB = -0.081
        CC = -1.01
    else:
        raise ValueError('halo definition not allowed')
  
    Mpivot = 2.e12
   
    return AA*np.power(mass/Mpivot, BB)*np.power(1.+redshift, CC)
       
       


