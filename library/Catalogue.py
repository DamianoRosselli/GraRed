import numpy as np
import pandas as pd
import warnings
import astropy.cosmology as asco
from astropy.cosmology import funcs
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy import costants as cst
from . import utils as ut
import abc


class Catalogue(abc.ABC): 
    """ General Catalogue Class """
    _object_type=''
	#class attrbute astropy cosmology

    def __init__(self,ra,dec,redshift,redshift_err=None, mass=None,radius=None,cosmo=asco.Planck18):
        
        self._objID=np.arange(len(ra))
        self._number_objects=len(ra)
        self._ra=ra
        self._dec=dec
        self._z=redshift

        if redshift_err is not None:
            self._redshift_err=redshift_err  
            
        if mass is not None:
            self._mass=mass
                
        if radius is not None:
            self._radius=radius

        set_cosmology(cosmo)
        self._comoving_distance=compute_comoving_distance(self._z,self._cosmology)
        self._comoving_transverse_distance=compute_transverse_comoving_distance(self._z,self._cosmology)
	

		
    def compute_comoving_distance(redshift,cosmology):
        d=cosmology.comoving_distance(redshift)
        return d

    def compute_transverse_comoving_distance(redshift,cosmology):
        d=cosmology.comoving_transverse_distance(redshift)
        return d
	
    def set_cosmology(self,cosmo):
     #set new Cosmology,  cosmo should be astropy cosmological model, default is Planck 18 """
	    self._cosmology = asco.cosmo
	
	
        
        	
    @property
    def object_type(self):
        if not hasattr(self,"_object_type"):
         	self._object_type = self._object_type
        return self._object_type
            
    @property
    def cosmology(self):
        return self._cosmology

    @property
    def objID(self):
        return self._objID

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def z(self):
        return self._z
    
    @property
    def mass(self):
        return self._mass

    @property
    def radius(self):
        return self._radius

    @property
    def comoving_distance(self):
        return self._comoving_distance

    @property
    def comoving_transverse_distance(self):
        return self._comoving_transverse_distance
        	
          
class Galaxies(Catalogue): 
    """ Galaxy Catalogue Object"""
    object_type='Galaxy' 

    def __init__(self,ra,dec,redshift,redshift_err=None,mass=None,radius=None,mag=None,cosmo=asco.Planck18):
            
        super().__init__(self,ra,dec,redshift,redshift_err,mass,radius,cosmo=asco.Planck18)
        
        if mag is not None:
            self._magnitude = mag
        
        self._data=pd.DataFrame({'RA':self._ra,
                                'Dec':self._dec,
                                'redshift': self._z,
                                'redshift_err': self._redshift_err,
                                'Radius':self._radius,
                                'Mass':self._mass,
                                'magnitude': self._magnitude,
                                'Dc':  self._comoving_distance,
                                'Dm': self._comoving_transverse_distance,
                                'ID': self._objID})

    #AION TO CALCULATE THE PARAMETRS THAT CORRECT FOR SURFACE BRIGHTNESS MODULATION
    
    
    def get(self,key=None,objid=None):
        if key is not None and objid is None:
            return self._data.loc[:,key]
        elif objid is not None and key is not None:
            s=self._data.loc[:,key]
            return s.iloc[objid]
        elif objid is not None and key is None:
            return self._data.iloc[objid]    
        else:
            raise ValueError('give a key or objID to select somenthing in self._data')

    @property
    def data(self):
        return self._data

    @property
    def magnitude(self):
        return self._magnitude

class MemberGalaxies(Catalogue): 
    """  Member Galaxy Catalogue Object"""
    _object_type='MemberGalaxy'  

    #add compute memership probability 
    	
class Cluster(Catalogue):
    """ Cluster Catalogue Object"""

    _object_type='Cluster'
	

    def __init__(self,ra,dec,redshift,redshift_err=None,mass=None,radius=None,concentration=None,fr0=None,
                compute_radius=False,compute_conc=False,convert_to_fr=False,is_crit=True,
                seed=1234,delta=500,cosmo=asco.Planck18,gravity='GR'):
            
        super().__init__(self,ra,dec,redshift,redshift_err,mass,radius,cosmo=asco.Planck18)
            
        set_Delta(delta)
        set_gravitymodel(gravity)
        self._is_crit=is_crit
        if gravity != 'GR' and fr0 is None:
            raise ValueError('set Fr0 if you are using f(R) gravity')
        elif gravity == 'GR' and fr0 is not None:
            raise ValueError('fr0 has to be None in GR')
        elif gravity != 'GR' and fr0 is not None:
            set_Fr0(fr0)

        if self._radius is None and compute_radius:
            self._radius=ut.halo_radius(self._mass,self._z,self._Delta,self._cosmology,self._is_crit)
                
        if self._mass is not None:
            print('if the masses are in logharitmic scale, please change it or the code will not work')
                
        if concentration is not None:
            self._concentration=concentration

        elif concentration is None and compute_conc:
            self._concentration=compute_concentration(seed)
               
        if  gravity == 'GR' and convert_to_fr:
            raise ValueError('please select gravity=f(R) if you want to convert quantities from GR to f(r)')
        if gravity != 'GR' and convert_to_fr:
            if self._mass is None:
                raise ValueError('mass to be converted are not in the Catalogue')
            if self._concentration is None:
                raise ValueError('concentration to be converted are not in the Catalogue')
            if self._radius is None:
                raise ValueError('radius to be converted are not in the Catalogue')
            else:
                print('attention convertion works only if M500 are given')
                self._concentration=self.convert_concentration_GRtofR(self._mass,self._z,self._Fr0,self._cosmology)
                self._mass=ut.convert_mass_GRtofR(self._mass,self._z,self._Fr0,self._cosmology)
                self._radius=ut.halo_radius(self._mass,self._z,self._Delta,self._cosmology,self._is_crit)

        self._data=pd.DataFrame({'RA':self._ra,
                                'Dec':self._dec,
                                'redshift': self._z,
                                'redshift_err': self._redshift_err,
                                'R_'+str(delta):self._radius,
                                'M_'+str(delta):self._mass,
                                'c_'+str(delta):self._concentration,
                                'Dc':  self._comoving_distance,
                                'Dm': self._comoving_transverse_distance,
                                'ID': self._objID})


                                          
    def convert_concentration_GRtofR(self,mass,redshift,fr0,cosmology):
        """ see mitchell et al 2021"""
        if self._Delta==200:
            y=ut.convert_concentration(mass,redshift,fr0,cosmology)
            return self.concentration*(10**y)
        
        elif self._Delta==500:
            y=ut.convert_concentration(mass,redshift,fr0,cosmology)
            return ut.compute_c500(self._z,self._mass,y)
            
            
    def compute_concentration(self,seed):            
        if self._Delta==200:
            c200=ut.Duffy_NFW_conc(self._mass,self._redshift,'200')
            return c200

        elif self._Delta==500:
            return ut.compute_c500(self._z,self._mass)

        else:
            raise ValueError('200 and 500 are the only value of Delta allowed')
    


    def get(self,key=None,objid=None):
        if key is not None and objid is None:
            return self._data.loc[:,key]
        elif objid is not None and key is not None:
            s=self._data.loc[:,key]
            return s.iloc[objid]
        elif objid is not None and key is None:
            return self._data.iloc[objid]    
        else:
            raise ValueError('give a key or objID to select somenthing in self._data')
            
            
    def compute_cluster_center(self, galaxy_data, radians=False, v_cut=2500., r_cut=1., **kwargs):      
        
        RAgal = [row.RA for row in galaxy_data.itertuples()]
        DECgal = [row.Dec for row in galaxy_data.itertuples()]
        Zgal = [row.redshift for row in galaxy_data.itertuples()]

        if not radians:
            self._data['RA rad']= np.radians(self._data.RA)
            self._data['Dec rad']= np.radians(self._data.RA)
            RAgal= np.radians(RAgal)
            DECgal= np.radians(DECgal)

        r=[ut.dist_between_2obj(row.RA,row.DEC,RAgal,DECgal,**kwargs) for row in self._data.itertuples()]
        v=[ut.vlos_center(row.redshift,Zgal) for row in self._data.itertuples()]

        dat=pd.DataFrame({'dist':r,
                          'V':v,
                          'IDclust': np.sort([self._objID for i in RAgal])})
                          #add ragal dec gal
        

        dat=dat[np.abs(dat.V)<=v_cut & dat.dist<=r_cut]
        
        count=dat.groupby('IDclust').count()
        mean_pos=dat.groupby('IDclust')[['RAgal','DECgal']].mean()




        decbo2=np.append(decbo[idv],dec[i])
        centre_ra=np.append(centre_ra,np.mean(rabo2))
        num=np.append(num,len(rabo2)) 
            

    

            
    def set_delta(self,delta):
	    self._Delta=delta
	
    def set_gravitymodel(self,grav):
	    self._GravityModel = grav

    def set_Fr0(self,fr0):
            self._Fr0=fr0
            
    @property
    def is_crit(self):
            return self._is_crit
        
    @property
    def Delta(self):
    	return self._Delta

    @property
    def number_objects(self):
        return self._number_objects
		
    @property 
    def GravityModel(self):
    	return self._GravityModel

    @property
    def concentration(self):
        return self._concentration

    @property
    def Fr0(self):
        if self._Fr0 is None:
            raise ValueError('Fr0 not set, if you are in f(R) gravity please set Fr0')
        else:
            return self._Fr0

    @property
    def data(self):
        return self._data

        

class Voids(Catalogue): 
    """ Voids Catalogue Object"""
    _object_type='Voids' 
	
    def __init__(self):
        raise NotImplementedError("Voids Catalogue is not yet available.") 
