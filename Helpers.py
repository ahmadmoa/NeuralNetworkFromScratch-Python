import numpy as np
import pandas as pd

def sigmoid(x):
    Sigmoid=1/(1+np.exp(-x))
    # DerivativeOf_Sigmoid=Sigmoid*(1-Sigmoid)
    return Sigmoid

def tanh(x):
    HyperbolicTangent_sigmoid =(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    # DerivativeOf_TanH=1-HyperbolicTangent_sigmoid**2
    return HyperbolicTangent_sigmoid

def get_Derivative(activation_value,activation_type):
    if (activation_type == 'Sigmoid'):
        return activation_value*(1-activation_value)
    else:
        return 1-activation_value**2

