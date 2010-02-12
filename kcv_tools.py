import mlpy

def kfold_splits(labels, k, rseed=0):
    return mlpy.kfold(labels.size, k, rseed)

def stratified_kfold_splits(labels, k, rseed=0):
    return mlpy.kfoldS(labels, k, rseed)