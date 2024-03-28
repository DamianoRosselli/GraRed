import numpy as np
import matplotlib as plt
from matplotlib import cm

def my_map(std_map,entries,under_cl,over_cl=None):

    return cm.get_cmap(std_map,entries).with_extremes(under=under_cl,over=over_cl)
    
