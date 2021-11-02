import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

#computes false positive rate and false negative rate given: predictions, targets and a threshold
def fp_fn_rates(y_pred, y_true, threshold):

    TP = FP = TN = FN = 0
    y_quant = np.zeros(len(y_true))
    
    for i in range(len(y_true)): 
        if y_pred[i] >= threshold:
            y_quant[i] = 1
        if y_true[i] == y_quant[i] == 1:
           TP += 1
        if y_quant[i] == 1 and y_true[i] == 0:
           FP += 1
        if y_true[i] == y_quant[i] == 0:
           TN += 1
        if y_quant[i] == 0 and y_true[i] == 1:
           FN += 1

    fp_rate = FP / (FP + TN)
    fn_rate = FN / (FN + TP)
    return(fp_rate, fn_rate)
    
#draws the det curve, plotting false positive rates against false negative rates
def det(fpr,fnr):
    
    fig,ax = plt.subplots(figsize=(10,10))
    plt.plot(fpr,fnr)

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    plt.xscale('log')
    plt.yscale('log')
    plt.axis([0.000001,1,0.000001,1])
    
    ax.set_ylabel('false negative rates')
    ax.set_xlabel('false positive rates')
    ax.set_title('DET curve')