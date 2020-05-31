import tensorflow as tf
from tensorflow.keras.layers import Layer
from math import pi

import numpy as np

from copy import deepcopy

tf.compat.v1.enable_eager_execution()

#want the hrrs to be generated the same
#tf.compat.v1.random.set_random_seed(100)

class Context(Layer):

  
  def __init__(self, num_hrrs, context_on=0, hardcoded_contexts=False):
    #print("init called")
    super(Context, self).__init__()
    
    self.num_hrrs = num_hrrs
    self.num_outputs = num_hrrs
    self.context_on = context_on
    self.hardcoded_contexts = hardcoded_contexts
    
    self.context_error_sums = [0 for x in range(self.num_hrrs)]
    self.selected_context_idx = 0
    
    
    
#------------------------------------------------------------------------------------------------------------------
  def get_hot_context(self):
    hot = self.kernel[0]
    return hot

#------------------------------------------------------------------------------------------------------------------
  def set_hot_context(self, hot_idx):
    """Left shift context list until the hot context is at the 0th index"""
    
    def left_shift_hot(shape, dtype=None):
        #print("orig weights:", self.orig_weights[0])    
    
        x = deepcopy(self.orig_weights[0])
    
        if hot_idx not in range(len(x)):
            raise ValueError("Hot context must be a valid index value — between 0 and " +  str(len(x)-1) + " inclusive")
    
        hot = hot_idx
    
        if hot_idx != 0: hot = abs(hot-len(x)) 
    
        y = [0 for x in range(len(x))]
    
        for i in range(len(x)):
            y[i] = x[ i - hot ]

        #print("y: ", y)
        return y
    
    #print("self.get_weigths()", self.get_weights() )
    self.set_weights(np.array([left_shift_hot([self.num_hrrs, self.inp_shp])]))
    
    #self.kernel = self.add_weight(name='context',
    #                              shape=[self.num_hrrs, self.inp_shp],
    #                              initializer=left_shift_hot,
    #                              trainable=False)        
#------------------------------------------------------------------------------------------------------------------
  def next_hot_context(self):
    
    
    #print("self.get_weigths()", self.get_weights() )
    self.set_weights(np.array([left_shift_hot([self.num_hrrs, self.inp_shp])]))
    
    #self.kernel = self.add_weight(name='context',
    #                              shape=[self.num_hrrs, self.inp_shp],
    #                              initializer=left_shift_hot,
    #                              trainable=False)       

#------------------------------------------------------------------------------------------------------------------

  def build(self, input_shape):
    #print("build called")
    
    #*******************************************************************************
    def hrr(length, normalized=True):
      length = int(length)      
      shp = int((length-1)/2)
    
      if normalized:    
        x = tf.random.uniform( shape = (shp,), minval = -pi, maxval = pi, dtype = tf.dtypes.float32, seed = 100, name = None )
        x = tf.cast(x, tf.complex64)
        
        if length % 2:
          x = tf.math.real( tf.signal.ifft( tf.concat([tf.ones(1, dtype="complex64"), tf.exp(1j*x), tf.exp(-1j*x[::-1])], axis=0)))
            
        else:  
          x = tf.math.real(tf.signal.ifft(tf.concat([tf.ones(1, dtype="complex64"),tf.exp(1j*x),tf.ones(1, dtype="complex64"),tf.exp(-1j*x[::-1])],axis=0)))
        
        
      else:        
        x = tf.random.normal( shape = (length,), mean=0.0, stddev=1.0/tf.sqrt(float(length)),dtype=tf.dtypes.float32,seed=100,name=None)
      
      return x
    #*******************************************************************************

    
    #*******************************************************************************
    def hrrs(length, n=1, normalized=True):
      return tf.stack([hrr(length, normalized) for x in range(n)], axis=0)  
    #*******************************************************************************
 

################################################################
#    s = hrrs(input_shape[-1], n=self.num_hrrs)
#    
#    self.context = tf.cast(s, tf.complex64)
#    self.hot_context = self.context[self.context_on]
################################################################


    #print("Is this a nested list [ [], [] ] ?:",hrrs(input_shape[-1], n=self.num_hrrs))

    # used in set_hot_context()
    self.inp_shp = int(input_shape[-1])

    
    #***********************************************************
    # Flag hardcoded_contexts=True
    #***********************************************************
    if self.hardcoded_contexts == True:
      
      #print("hardcoed_contexts=True")
    
      # create hardcoded untrainable weights ie. hrr placeholders for debugging
      # If input_shape[-1] = 2
      # and self.num_hrrs = 4
      # Will produce:  [ [1,1], [-2, -2], [3,3], [-4,-4] ]
      x = [ [ 1*x for y in range(1,1 + input_shape[-1]) ] for x in range(1,1+ self.num_hrrs) ]  
      for lis in range(len(x)):
        for ele in range(len(x[lis])):
          if lis % 2:
            x[lis][ele] *= -1
    
      self.kernel = self.add_weight(name='context',
                                  shape=[self.num_hrrs, int(input_shape[-1])],
                                  initializer=tf.keras.initializers.Constant(x),
                                  trainable=True)
        
    #***********************************************************
    # Flag hardcoded_contexts=False
    #***********************************************************
    else:
      
      #print("hardcoed_contexts=False")

      def init_hrrs(shape, dtype=None):
        x = hrrs(input_shape[-1], n=self.num_hrrs)
        #print("hrrs inside self.kernel", x) 
        #print("type of hrrs inside self.kernel", type(x))
        return x
    
      self.kernel = self.add_weight(name='context',
                                  shape=[self.num_hrrs, int(input_shape[-1])],
                                  initializer=init_hrrs,
                                  trainable=False)
    
    
    self.hot_context = self.kernel[self.context_on]
    
    self.orig_weights = self.get_weights()
    



#------------------------------------------------------------------------------------------------------------------
  def call(self, inputs):
    #print("Call invoked")
    
    def circ_conv():
      #print("Convolution invoked")
      inp = tf.cast(inputs, tf.complex64)
      #print("inside circ_conv() -- inp:", inp)
    
      #con = tf.cast(self.get_hot_context(), tf.complex64)
      #print("inside circ_conv() -- inp:", inp)
          
      #return tf.math.real(tf.signal.ifft(tf.signal.fft(inp)*tf.signal.fft(self.hot_context)))  
      return tf.math.real(tf.signal.ifft(tf.signal.fft(inp)*tf.signal.fft(tf.cast(self.kernel[0], tf.complex64))))  


    return circ_conv()

