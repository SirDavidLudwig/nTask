import numpy as np
import tensorflow as tf

def test(model):

    raw_task_preds = []
    rounded_task_preds = []
    
    data = [  
             np.array([[-1, -1]]),
             np.array([[-1, 1]]),
             np.array([[1, -1]]),
             np.array([[1, 1]]),
       ]
    
    for x in data:

        #print( model(x, training=False) )
        #print( tf.math.round(model(x, training=False)).numpy() )
        
        raw_task_preds.append( model(x, training=False).numpy() )
        rounded_task_preds.append( tf.math.round(model(x, training=False)).numpy() )
        
    return raw_task_preds, rounded_task_preds