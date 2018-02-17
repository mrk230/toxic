from sklearn.metrics import roc_auc_score
from metrics import columnwise_auc

class ROC_Eval(Callback):
    def __init__(self, X_val, y_val):
        super(Callback, self).__init__()
        
        self.X_val= X_val
        self.y_val = y_val
        
    def on_train_begin(self, logs={}):
        self.aucs = []
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        score = columnwise_auc(self.y_val, y_pred)
        self.aucs.append(score)
        print("\n ROC-AUC - score: {:.6f}".format(score))
        return

def toggle_train(model):
    """
    Toggles the trainable param on a keras model (outside of input layer)
    Use this for when we are trying to train just our embedding layer
    
    Input:
        model - a keras model
        
    Output:
        model - the changed keras model
    """
    for layer in model.layers[1:]:
        if layer.trainable == True:
            layer.trainable = False
#             print(layer)
#             print("now false")
#             print("")
        elif layer.trainable == False:
            layer.trainable = True
            print(layer)
            print("now true")
            print("")
    
    return model
                 
def all_layers_train(model):
    """
    Makes all layers trainable
    Use this when trying to train the entire model
    
    Input:
        model - a keras model
    
    Output:
        model - the changed keras model
    """

    for layer in model.layers[1:]:
        layer.trainable = True
#         print(layer)
#         print("now True")
#         print("")
    
    return model
