import numpy as np
from data_tools import center
    
def classification_error(labels, predicted):
    difference = (np.sign(labels) != np.sign(predicted))
    return labels[difference].size / float(labels.size)
    
def balanced_classification_error(labels, predicted):
    balance_factors = np.abs(center(labels)[0])
   
    errors = (np.sign(labels) != np.sign(predicted)) * balance_factors
    return errors.sum() / float(labels.size)
    
def regression_error(labels, predicted):
    norm = np.linalg.norm(labels - predicted, 2)
    return (norm * norm) / float(labels.size)