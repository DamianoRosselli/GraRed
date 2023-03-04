import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import CosmoBolognaLib as cbl
from CosmoBolognaLib  import StringVector as sv
import astropy.cosmology as asco
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import funcs
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck18
from scipy.integrate import quad

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
cosmo=cbl.Cosmology(cbl.CosmologicalModel__Planck18_)
hhh=cosmo.hh()
#print(hhh)
cosmo.set_unit(True)
area=6800
voltot=cosmo.Volume(0.05,0.5,area)


file_cat='./DR16.flux.SDSS.0.05<z<0.5.zerr6e-4.dat'
z1,flux1=np.genfromtxt(file_cat,skip_header=1,unpack=True)
file_cat='./DR16.flux.BOSS.0.05<z<0.5.zerr6e-4.dat'
z2,flux2=np.genfromtxt(file_cat,skip_header=1,unpack=True)
z=np.append(z1,z2)
#flux=np.append(flux1,flux2)
#idf=np.where(flux>0)
#z=z[idf]
#flux=flux[idf]
#mg=22.5 - (2.5*np.log10(flux))
#mod_dist=Planck18.distmod(z).value
#Mag=mg-mod_dist+(2.5*np.log10(1.1))-(5*np.log10(hhh))
#idmag=np.where(Mag>-26)
#Mag=Mag[idmag]
#z=z[idmag]

histz,bins=np.histogram(z,1000)
mean=np.empty(len(histz),dtype=float)
dndz=np.empty(len(histz),dtype=float)
binsize=np.empty(len(histz),dtype=float)
for i in range(len(histz)):
    mean[i]=(bins[i]+bins[i+1])*0.5
    dndz[i]=histz[i]/(bins[i+1]-bins[i])
    binsize[i]=np.abs(bins[i+1]-bins[i])


    
dndz=dndz/voltot
dn=histz*(mean**2)
#dn=dn/voltot

plt.figure()
plt.plot(mean,dndz)
plt.title('dndz')
#plt.show()

plt.figure()
plt.plot(mean,dn)
plt.title('dn')

def mod_dist(x):
    return 5*np.log10(cosmo.D_L(x))+25

def lum_func(x):
   
    phi=0.0093#*(hhh**(3))
    M=-20.71#+(5*np.log10(hhh))
    a=-1.26
    expo=np.exp(-10**(0.4*(M-x)))
    power=10**(0.4*(M-x)*(a+1))
    return 0.4*np.log(10)*phi*power*expo

def cumul_lum(mag):
    return quad(lum_func,-100,mag)[0]

vec_cum=np.vectorize(cumul_lum)


mod_dist1=np.asarray([mod_dist(i) for i in mean])
Mag1=22.29-mod_dist1-(-2.5*np.log10(1.1))#+(5*np.log10(hhh))

#nz=vec_cum(Mag1)*(yyy**2)
deltaz=lum_func(Mag1)/vec_cum(Mag1) 

a=np.sum(deltaz*dndz*binsize)/np.sum(dndz*binsize)
print(a)

b=np.sum(deltaz*dn*binsize)/np.sum(dn*binsize)
print(b)

plt.figure(figsize=(10,8))
plt.plot(mean,deltaz,color='r',label='delta(z)')
#plt.plot(yyy,nz,label='z^2 n(z)')
plt.xlim(0.05,0.5)
plt.xlabel('z',fontsize=15)
plt.ylabel(r'$\delta(z)$',fontsize=15)
#plt.ylim(0,6)
#plt.legend()
plt.savefig('./immagini/surf.mod.22.29.png')
plt.show()

def maglim(z):
    mod_dist1=mod_dist(z)
    return 22.29-mod_dist1+(2.5*np.log10(1.1))#+(5*np.log10(hhh))
   
   
def maglimup(z):
    mod_dist=Planck18.distmod(z).value    
    return 13.93-mod_dist-(5*np.log10(hhh))+(2.5*np.log10(1.1))
   

def den(z):
    return (z**2) * cumul_lum(maglim(z))

def num(z):
    return (z**2) *  lum_func(maglim(z))



def integral(z1,z2):
    nume=quad(num,z1,z2)[0]
    deno=quad(den,z1,z2)[0]
    return nume/deno

#print(integral(0.016,0.667))
print(integral(0.05,0.5))
#print(integral(0.1,0.4))
#print(integral(0.05,0.4))
#print(integral(0.05,0.3))
#print(integral(0.05,0.6))




import math
import operator
import emcee
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import CosmoBolognaLib as cbl
from CosmoBolognaLib  import StringVector as sv
import astropy.cosmology as asco
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import funcs
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck18
from scipy.integrate import quad,dblquad,nquad
from scipy.optimize import curve_fit
from astropy import constants as const
from tqdm import tqdm

cosmology=cbl.Cosmology(cbl.CosmologicalModel__Planck18_)
cosmology.set_unit(False)

file_data='./data/hope/dataformodel.whl15.match.mixsdssboss.meanallmembr>2.ramean.mselect.z<0.5.RL>10.4mem.dat' #file velocity selected galaxies
zn,r500n,logm500n,c500n=np.genfromtxt(file_data,skip_header=1,unpack=True)
m500n=10**(logm500n)
rho_cn=Planck18.critical_density(zn).to('kg/m3')
#deltaz=1.29 #to z<0.4
#deltaz=1.3144 #to z<0.6
deltaz=0.5159

c500_median=np.median(c500n)
rho_median=np.median(rho_cn)
M500_min=np.min(m500n)
M500_max=np.max(m500n)
rho_msun=np.asarray([cosmology.rho_crit(zn[i]) for i in range(len(zn))])
rho_m_msun=np.median(rho_msun)
rho_std_msun=np.std(rho_msun)

#binsm=np.logspace(np.min(np.log10(m500n)),np.max(np.log10(m500n)),20)
#histm,bins=np.histogram(m500n,binsm)

binsm3=np.logspace(np.min(np.log10(m500n)),14.05,30)
binsm2=np.logspace(14.35,np.max(np.log10(m500n)),24)

histm3,bins3=np.histogram(m500n,binsm3)

mean_bin3=np.empty(len(histm3),dtype=float)
hist_dm3=np.empty(len(histm3),dtype=float)
error3=np.empty(len(histm3),dtype=float)
for i in range(len(histm3)):
    hist_dm3[i]=histm3[i]/((bins3[i+1]-bins3[i]))
    mean_bin3[i]=(bins3[i]+bins3[i+1])*0.5
    error3[i]=np.sqrt(histm3[i])/((bins3[i+1]-bins3[i]))

error3=error3/(hist_dm3*np.log(10))
logbin3=np.log10(mean_bin3)
loghist3=np.log10(hist_dm3)

histm2,bins2=np.histogram(m500n,binsm2)

mean_bin2=np.empty(len(histm2),dtype=float)
hist_dm2=np.empty(len(histm2),dtype=float)
error2=np.empty(len(histm2),dtype=float)
for i in range(len(histm2)):
    hist_dm2[i]=histm2[i]/((bins2[i+1]-bins2[i]))
    mean_bin2[i]=(bins2[i]+bins2[i+1])*0.5
    error2[i]=np.sqrt(histm2[i])/((bins2[i+1]-bins2[i]))

error2=error2/(hist_dm2*np.log(10))
logbin2=np.log10(mean_bin2)
loghist2=np.log10(hist_dm2)


zz = np.polyfit(logbin3,loghist3, 3)
f = np.poly1d(zz)
print(f)


def lin4(x,a,b,c,d,e):
    return a + b*x +c*(x**2) +d*(x**3)+e*(x**4)

def lin3(x,a,b,c):
    return a + b*x +c*(x**2)

def lin2(x,a,b,c):
    return a + b*x +c*(x**2) 



init2=[-591,82,-3]
popt2,pcov2=curve_fit(lin2,logbin2,loghist2,init2,sigma=error2,absolute_sigma=False,maxfev=2000)

label=['a','b','c','d','e']
error_fit=np.empty(len(popt2),dtype=float)
for i in range(len(popt2)):
    error_fit[i]=np.sqrt(pcov2[i,i])
    print(label[i],':',popt2[i],'pm',np.sqrt(pcov2[i,i]))
chi2=np.sum((loghist2-lin2(logbin2,*popt2))**2/error2**2)   
print(chi2/21.)

plt.figure(figsize=(10,8))
plt.plot(logbin2,lin2(logbin2,*popt2))
plt.errorbar(logbin2,loghist2,error2)

zeros=np.zeros(len(logbin2))
plt.figure(figsize=(10,8))
plt.plot(logbin2,zeros)
plt.errorbar(logbin2,loghist2-lin2(logbin2,*popt2),error2)
plt.show()



init3=[-693,98,3.5]
popt3,pcov3=curve_fit(lin3,logbin3,loghist3,init3,sigma=error3,absolute_sigma=False,maxfev=2000)

label=['a','b','c','d','e']
error_fit=np.empty(len(popt3),dtype=float)
for i in range(len(popt3)):
    error_fit[i]=np.sqrt(pcov3[i,i])
    print(label[i],':',popt3[i],'pm',np.sqrt(pcov3[i,i]))
chi2=np.sum((loghist3-lin3(logbin3,*popt3))**2/error3**2)   
print(chi2/27.)

plt.figure(figsize=(10,8))
plt.plot(logbin3,lin3(logbin3,*popt3))
plt.errorbar(logbin3,loghist3,error3)

zeros=np.zeros(len(logbin3))
plt.figure(figsize=(10,8))
plt.plot(logbin3,zeros)
plt.errorbar(logbin3,loghist3-lin3(logbin3,*popt3),error3)
plt.show()


def g_c5(c):
    return 1./(np.log(1+c)-(c/(1+c)))



def NFW_2d(x,c,R) :#funzione da integrare per avere profilo NFW 2D
    return 1./(((1+c*x)**2)*np.sqrt(x**2-R**2))

def potential(x,c,R): #funzione da integrare per avere il potenziale gravitazionale 
    return (-c + (np.log(1+c*x)/x))*NFW_2d(x,c,R)

def NUM2(x):
    return x*np.power(10,lin2(np.log10(x),*popt2))

def NUM3(x):
    return x*np.power(10,lin3(np.log10(x),*popt3))

def DEN2(x):
    return np.power(x,1./3.)*np.power(10,lin2(np.log10(x),*popt2))

def DEN3(x):
    return np.power(x,1./3.)*np.power(10,lin3(np.log10(x),*popt3))

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

m_lim_low=10**(14.05)
m_lim_up=10**(14.35)
def fit_func(rr,a):
    const_fac=(4*np.pi/3)*np.power(3/(4*np.pi),2/3)*a*const.G*np.power(const.M_sun,2./3.)/(const.c)


    num_r=vec_pot_int(c500_median,rr)
    num_m1=quad(NUM3,M500_min,m_lim_low)[0]
    num_m2=quad(NUM2,m_lim_up,M500_max)[0]
    num_m=num_m1+num_m2
    #num_m=quad(NUM,M500_min,M500_max)[0]
    num=num_r*num_m

    den_r=vec_nfw_int(c500_median,rr)
    den_m1=quad(DEN3,M500_min,m_lim_low)[0]
    den_m2=quad(DEN2,m_lim_up,M500_max)[0]
    den_m=den_m1+den_m2
    #den_m=quad(DEN,M500_min,M500_max)[0]
    den=den_m*den_r

    redshift=(const_fac*(num/den)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    redshift=redshift.value

    #model TD
    num_r=vec_td_int(c500_median,rr)
    num_m1=quad(NUM3,M500_min,m_lim_low)[0]
    num_m2=quad(NUM2,m_lim_up,M500_max)[0]
    num_m=num_m1+num_m2
    #num_m=quad(NUM,M500_min,M500_max)[0]
    num=num_r*num_m

    den_r=vec_nfw_int(c500_median,rr)
    den_m1=quad(DEN3,M500_min,m_lim_low)[0]
    den_m2=quad(DEN2,m_lim_up,M500_max)[0]
    den_m=den_m1+den_m2
    #den_m=quad(DEN,M500_min,M500_max)[0]
    den=den_m*den_r

    redshift_td=((3./2.)*const_fac*(num/den)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    redshift_td=redshift_td.value

    finale=redshift+(redshift_td*(4-(10*deltaz))/3)
    return finale



bin_dist=np.array([0.4930291580287251 ,1.495565328569689, 2.503292134133705 ,3.511846551638351])
err_bin_dist=np.array([0.28833556083756528 ,0.28145382259167817, 0.2908587469997078 ,0.28876764753570244])


vmm=np.asarray([ -4.65115, -11.1562 , -10.8854 , -17.7519 ])
evmm=np.asarray([3.57436, 6.51726, 7.73587, 8.38579])#vmean

vmc=[ -4.88197 , -9.52801 ,-18.8464,  -15.8996 ]
evmc=[3.62047, 6.28544, 7.7041 , 8.21468]#raclust

initred=[1]
poptred,pcovred=curve_fit(fit_func,bin_dist,vmm,initred,sigma=evmm,absolute_sigma=False,bounds=(0,10),maxfev=2000)

print(poptred,'pm',np.sqrt(pcovred))
    
chi2=(np.sum((vmm-fit_func(bin_dist,poptred))**2/(evmm**2)))/3.

xx=np.linspace(0.001,4,100)
plt.figure(figsize=(20,10))
plt.plot(xx,fit_func(xx,poptred),label='Fit')
plt.plot(xx,fit_func(xx,1.0),color='r',linestyle='--',label='GR')
plt.plot(xx,fit_func(xx,4./3.),color='black',linestyle='--',label='f(R)')
plt.fill_between(xx,fit_func(xx,poptred+np.sqrt(pcovred[0,0])),fit_func(xx,poptred-np.sqrt(pcovred[0,0])),alpha=0.2)
plt.errorbar(bin_dist,vmm,evmm,err_bin_dist,color='g',marker='o',linestyle='',label='a = 0.62 pm 0.17 ; chi2/ndof=0.66')
plt.errorbar(bin_dist,vmc,evmc,err_bin_dist,color='purple',marker='o',linestyle='',label='a = 0.62 pm 0.17 ; chi2/ndof=0.66')
plt.xlim(np.min(xx),np.max(xx))
plt.legend()
plt.title('Fit curve_fit, minimo chi2')
#plt.savefig('./immagini/tesi/whl15.match.mixsdssboss.meanallmembr>2.ramean.mselect.z<0.5.RL>10.4mem.curvefit.png')
plt.show()

aaa=np.linspace(-0.5,1.5,100)
chi2a=np.empty(len(aaa),dtype=float)
for i in range(len(aaa)):
    chi2a[i]=(np.sum((vmm-fit_func(bin_dist,aaa[i]))**2/(evmm**2)))/3.

print(np.min(chi2a))
one=np.ones(len(aaa))    
plt.figure(figsize=(20,10))
plt.plot(aaa,chi2a,label='chi2/ndof min = 0.66')
plt.plot(aaa,one,color='r',linestyle='--')
plt.xlim(np.min(aaa),np.max(aaa))
plt.legend()
plt.title('test chi2')
#plt.savefig('./immagini/tesi/whl15.match.mixsdssboss.meanallmembr>2.ramean.mselect.z<0.5.RL>10.4mem.testchi2.png')
plt.show()

##### markov-chain #####


def  lnPrior( theta ) :
    a = theta
    if  a > 0 :
        return  0.0 
    else:
        return -np.inf
    
    
    
def lnLike(theta,x,y,yerr):
    a=theta
    s2=yerr**2
    mod=fit_func(x,theta)
    return -0.5*np.sum(np.log(2*np.pi*s2)+ (((y-mod)**2)/s2))

#def lnLike(theta,x):
    #a=theta
    #mod=fit_func(x,theta)
    #return np.sum(np.log(mod))


def lnPost(theta,x,y,yerr):
    
    prior=lnPrior(theta)
    
    if not np.isfinite(prior):
        return -np.inf
   
    return prior + lnLike(theta,x,y,yerr)

npar=1
start=np.asarray([1.0])
xx=np.linspace(0.001,4,1000)
plt.figure(figsize=(10,10))
plt.title('init')
plt.errorbar(bin_dist,vmm,evmm,err_bin_dist,color='g',marker='o',linestyle='',label='zclust<0.5')
plt.errorbar(bin_dist,vmc,evmc,err_bin_dist,color='purple',marker='o',linestyle='',label='zclust<0.5')
plt.plot(xx,fit_func(xx,start),color='r')
plt.xlim(np.min(xx),np.max(xx))
plt.show()

if np.isfinite(lnPrior(start)):
    print('Start value ok!!')


nwalk=100

pos0=[ start + 1.e-3 * np.random.rand(npar) for i in range(nwalk)]


for i in range(nwalk):
    p=lnPrior(pos0[i])
    if not np.isfinite(p):
        print('Start chain value out of range !!!!')
        
        break
#setting sampler 

sampler=emcee.EnsembleSampler(nwalk,npar,lnPost,args=(bin_dist,vmm,evmm))

#run MCMC

pos,prob,rstate=sampler.run_mcmc(pos0,3000,progress=True)
tau = sampler.get_autocorr_time()
print('autocorr time:',tau)
print('mean acc. fraction:',np.mean(sampler.acceptance_fraction))


samples = sampler.get_chain()

plt.figure(figsize=(10,6))
plt.xlim(0,len(samples))
plt.xlabel('step')
plt.ylabel('a')
plt.plot(samples[:, :, 0], "k", alpha=0.3)
plt.show()



from scipy.stats import ks_2samp


flat_samples = sampler.get_chain(discard=500, thin=1, flat=True)


mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
q = np.diff(mcmc)
fit_res=mcmc[1]
fit_low=mcmc[1]-q[0]
fit_up=mcmc[1]+q[1]
print('a:',fit_res)
print('error up',q[1])
print('error low',q[0])


    
chi2=np.sum((vmm-(fit_func(bin_dist,fit_res)))**2/(evmm)**2)
chi2=chi2/(len(vmm)-npar)
print('chi2 redux:', chi2)

def model_FR():
    fin,a,b,c,d=model_GR()
    FR=(a*4./3.)+((2-(5*deltaz))*8*d/9.)
    return FR


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


def f3(y,x,c,R,rho):
    return  NUM3(x)*potential_dgp(y,c,x,R,rho)


def numdgp_int3(c,R,rho):  
    return nquad(f3,[[R,np.inf],[M500_min,m_lim_low]], args=(c,R,rho))[0]


def ftd3(y,x,c,R,rho):
    return NUM3(x)*TD(y,c,R)*g_dgp(y,c,x,R,rho)

def tddgp_int3(c,R,rho):  
    return nquad(ftd3,[[R,np.inf],[M500_min,m_lim_low]], args=(c,R,rho))[0]

vec_numdgp3=np.vectorize(numdgp_int3)
vec_tddgp3=np.vectorize(tddgp_int3)

def f2(y,x,c,R,rho):
    return  NUM2(x)*potential_dgp(y,c,x,R,rho)


def numdgp_int2(c,R,rho):  
    return nquad(f2,[[R,np.inf],[m_lim_up,M500_max]], args=(c,R,rho))[0]


def ftd2(y,x,c,R,rho):
    return NUM2(x)*TD(y,c,R)*g_dgp(y,c,x,R,rho)

def tddgp_int2(c,R,rho):  
    return nquad(ftd2,[[R,np.inf],[m_lim_up,M500_max]], args=(c,R,rho))[0]

vec_numdgp2=np.vectorize(numdgp_int2)
vec_tddgp2=np.vectorize(tddgp_int2)

rr=np.linspace(0.001,4,100)
#rimetti a posto unità di misura
const_fac_dgp=0.96*(4*np.pi/3)*np.power(3/(4*np.pi),2/3)*const.G*np.power(const.M_sun,2./3.)/(const.c)

def model_DGP():
    num3=vec_numdgp3(c500_median,rr,rho_m_msun)
    num2=vec_numdgp2(c500_median,rr,rho_m_msun)
    nu=num3+num2
    
    den_r=vec_nfw_int(c500_median,rr)
    den_m3=quad(DEN3,M500_min,m_lim_low)[0]
    den_m2=quad(DEN2,m_lim_up,M500_max)[0]
    den_m=den_m3+den_m2
    den=den_m*den_r
    nude=nu/den
   
    

    redshift=(const_fac_dgp*(nude)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    #model TD
    num3=vec_tddgp3(c500_median,rr,rho_m_msun)
    num2=vec_tddgp2(c500_median,rr,rho_m_msun)
    nu=num2+num3

    den_r=vec_nfw_int(c500_median,rr)
    den_m3=quad(DEN3,M500_min,m_lim_low)[0]
    den_m2=quad(DEN2,m_lim_up,M500_max)[0]
    den_m=den_m3+den_m2
    den=den_m*den_r
    nude=nu/den
    

    redshift_td=((3./2.)*const_fac_dgp*(nude)*g_c5(c500_median)*np.power(500*rho_median,1./3.)).to('km/s')
    finale=redshift+((2-(5*deltaz))*2*redshift_td/3.)
    return redshift,redshift_td,finale
    


dgpnocorr,dgptd,dgp=model_DGP()
      
plt.figure(figsize=(14,10))
plt.errorbar(bin_dist,vmm,evmm,err_bin_dist,color='purple',marker='o',linestyle='',label='a= 0.81 pm 0.25; chi2/ndof = 0.33 ')
#plt.errorbar(bin_dist,vmc,evmc,err_bin_dist,color='purple',marker='o',linestyle='',label='a= 0.62 pm 0.21; chi2/ndof = 0.66 ')
plt.plot(xx,fit_func(xx,fit_res),label='FIT')
plt.fill_between(xx,fit_func(xx,fit_up),fit_func(xx,fit_low),alpha=0.2)
plt.plot(xx,fit_func(xx,1),color='black',linestyle='--',label='GR')
plt.plot(xx,fit_func(xx,4./3.),color='r',linestyle='--',label='f(R)')
plt.plot(rr,dgp,color='g',linestyle='--',label='DGP')
plt.xlim(np.min(xx),np.max(xx))
plt.legend()
plt.title('Fit chain, gaussian likelihood')
plt.savefig('./immagini/tesi/whl15fit.massselect.chain.png')
plt.show()


