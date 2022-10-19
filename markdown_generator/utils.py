import numpy as np 
import math 

def isnan(x):
    if type(x) is str: 
        return x is np.nan
    else: 
        return math.isnan(x)

def isvalid(x): 
    if x == "": 
        return False 
    else: 
        return not isnan(x)