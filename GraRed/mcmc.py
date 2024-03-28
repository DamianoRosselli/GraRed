import numpy as np
import math
import itertools
from GraRed import Fit_bckg_corr as fbc
import emcee
from scipy.optimize import minimize

#model with two gaussian functions summed
def model(theta,x,a):

    #free parameters of the gaussians
    epsilon,sigma1,sigma2,delta_mean= theta

    if a == "2Gauss":
        return (epsilon/np.sqrt(2*math.pi*sigma1**2))*np.exp(-(x-delta_mean)**2/(2*sigma1**2))+ \
       ((1-epsilon)/np.sqrt(2*math.pi*sigma2**2))*np.exp(-(x-delta_mean)**2/(2*sigma2**2))
    else:
        raise ValueError("at the moment, we have only 2Gauss model")

#define the logarithm of a generic gaussian likelihood
def log_likelihood(theta,x,y, yerr,model_name):
    
    mod=model(theta,x,model_name)
    return -0.5 * np.sum((y - mod) ** 2 / yerr**2 + np.log(2*math.pi*yerr**2))


#define a prior for the 2gaussian model
def log_prior(theta,params):
    
    epsilon,sigma1,sigma2,delta_mean= theta
    epsilon_min,epsilon_max,sigma_min,sigma_max,delta_mean_range=params

    #define a flat prior 
    if epsilon_min < epsilon < epsilon_max and sigma_min <sigma1< sigma_max and sigma_min <sigma2< sigma_max and sigma1 < sigma2 and -abs(delta_mean) < delta_mean_range:
        return 0.0
    
    return -np.inf


#posterior definition
def log_probability(theta,params, x, y, yerr, model_name):
    
    lp = log_prior(theta,params)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta,x,y, yerr, model_name)


def mcmc(init_param,x,h,herr,model_name,params,walker_param,filename):
	
    seed,step_walker,n_walker,chain_length,discard,thin=walker_param
    #initial guess for the free parameters
    np.random.seed(seed)
    nll = lambda *args: -log_likelihood(*args) 
    #maximize the likelihood
    soln = minimize(nll, init_param, args=(x, h, herr,model_name))
    #parameters which maximize the likelihood
    eps_ml, s1_ml, s2_ml, dm_ml= soln.x
    
    #walker position
    pos = soln.x + step_walker * np.random.randn(n_walker, len(init_param))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(params,x, h, herr,model_name))
    sampler.run_mcmc(pos, chain_length, progress=True)
    sampler.get_autocorr_time()
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    backend = emcee.backends.HDFBackend(filename+".h5")

    return samples,flat_samples

def red_chisq(x,h,herr,model_name,params):
    chisq=np.sum((model(params,x,model_name)-h)**2/herr**2)
    return chisq/(len(h)-len(params))
    
    
