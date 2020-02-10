from sklearn.metrics import roc_curve 

def createROC(Y, pred):
    fpr, tpr, threshold = roc_curve(Y, pred)
    
    plt.figure()
    lw = 2
    plt.plot()