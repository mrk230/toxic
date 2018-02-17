from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def accuracy(y_test, probs):
    probs_acc = probs >= 0.5
    probs_acc = probs_acc.astype(int)
    acc = accuracy_score(y_test, probs_acc)
    
    return acc

def columnwise_auc(y_test, probs):
    y_0 = [x[0] for x in y_test]
    p_0 = [x[0] for x in probs]
    auc_0 = roc_auc_score(y_0, p_0)
    
    y_1 = [x[1] for x in y_test]
    p_1 = [x[1] for x in probs]
    auc_1 = roc_auc_score(y_1, p_1)
    
    y_2 = [x[2] for x in y_test]
    p_2 = [x[2] for x in probs]
    auc_2 = roc_auc_score(y_2, p_2)
    
    y_3 = [x[3] for x in y_test]
    p_3 = [x[3] for x in probs]
    auc_3 = roc_auc_score(y_3, p_3)
    
    y_4 = [x[4] for x in y_test]
    p_4 = [x[4] for x in probs]
    auc_4 = roc_auc_score(y_4, p_4)
    
    y_5 = [x[5] for x in y_test]
    p_5 = [x[5] for x in probs]
    auc_5 = roc_auc_score(y_5, p_5)
    
    mean_col_auc = (auc_0 + auc_1 + auc_2 + auc_3 + auc_4 + auc_5) / 6.0
    
    return mean_col_auc