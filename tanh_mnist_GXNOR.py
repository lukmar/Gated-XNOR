# The basic code framework is based on the BinaryNet (https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/Train-time/binary_net.py)
# We mainly modify the gradient calculation (e.g. discrete_grads function) and neuronal activition (e.g. discrete_neuron_3states) for network training. 
# And we save the best parameters for searching a better result.
# For multilevel extension, you can simply modify the activation function and the N parameter for weight.
# Please cite our paper if you use this code: https://arxiv.org/pdf/1705.09283.pdf

from __future__ import print_function

import sys
import os
import time
import getopt
import argparse

import numpy as np
np.random.seed(1234)  # for reproducibility


import theano
import theano.tensor as T
# specifying the gpu to use
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0') 

import lasagne

import cPickle as pickle
import gzip

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict

import time

import numpy as np

from theano.ifelse import ifelse

import matplotlib.pyplot as plt #for drawing
import scipy.io as scio
from numpy import random
from numpy import multiply

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex, BinaryScalarOp, upgrade_to_float
from theano.tensor.elemwise import Elemwise
from itertools import izip
class round_custom(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round_scalar = round_custom(same_out_nocomplex, name='round_var')
round_var = Elemwise(round_scalar)


def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)	

def discrete_neuron_3states(x): #discrete activation with three states
     return T.cast(round_var(hard_sigmoid(2*(x-1))+hard_sigmoid(2*(x+1))-1 ),theano.config.floatX)

# This class extends the Lasagne DenseLayer to support Probabilistic Discretization of Weights
class DenseLayer(lasagne.layers.DenseLayer): # H determines the range of the weights [-H, H], and N determines the state number in discrete weight space of 2^N+1
    
    def __init__(self, incoming, num_units, 
        discrete = True, H=1.,N=1., **kwargs): 
        
        self.discrete = discrete
        
        self.H = H
        self.N = N
        
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        if self.discrete:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the discrete tag to weights            
            self.params[self.W]=set(['discrete'])
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
    

# This class extends the Lasagne Conv2DLayer to support Probabilistic Discretization of Weights
class Conv2DLayer(lasagne.layers.Conv2DLayer): # H determines the range of the weights [-H, H], and N determines the state number in discrete weight space of 2^N+1
    
    def __init__(self, incoming, num_filters, filter_size,
        discrete = True, H=1.,N=1.,**kwargs):
        
        self.discrete = discrete
        
        self.H = H 
        self.N = N 
                  
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.discrete:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the discrete tag to weights            
            self.params[self.W]=set(['discrete'])
            
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)   

#fine tuning of weight element to locate at the neareast 2^N+1 descrete states in [-H, H] 
def weight_tune(W,l_limit,r_limit):
    global N
    state_index = T.cast(T.round((W-l_limit)/(r_limit-l_limit)*pow(2,N)),theano.config.floatX)
    W = state_index/pow(2,N)*(r_limit-l_limit) + l_limit

    return W

# def lfsr16(state):#xorshift
#     s = np.uint16(state) ^ (np.uint16(state) >> np.uint16(7))
#     s ^= np.uint16(s) << np.uint16(9)
#     s ^= np.uint16(s) >> np.uint16(13)
#     # print(state)
#     # return T.cast(state, theano.config.floatX)
#     return np.uint16(s)

global taps
taps = np.array([   
    [],
    [],
    [],
    [3,2],          #3
    [4,3],          #4
    [5,3],          #5
    [6,5],          #6
    [7,6],          #7
    # [8,6,5,1],      #8
    [8,6,5,4],      #8  original
    [9,5],          #9
    [10,7],         #10
    [11,9],         #11
    [12,6,4,1],     #12
    [13,4,3,1],     #13
    [14,5,3,1],     #14
    [15,14],        #15
    [16,15,13,4],   #16
    [17,14],        #17
    [18,11],        #18
    [19,6,2,1],     #19
    [20,17],        #20
    [21,19],        #21
    [22,21],        #22
    [23,18],        #23
    [24,23,22,17],  #24
    [25,22],        #25
    [26,6,2,1],     #26
    [27,5,2,1],     #27
    [28,25],        #28
    [29,27],        #29
    [30,6,4,1],     #30
    [31,28],        #31
    [32,22,2,1]     #32
]) 

def lfsr_n(state, n):#fibonnaci
    global taps
    if len(taps[n]) == 0:
        print("Error for n = " + str(n))
        return -1

    s = ((np.uint32(state) >> np.uint32(taps[n][0])) ^ (np.uint32(state) >> np.uint32(taps[n][1])))
    if len(taps[n]) == 4:
        s ^= np.uint32(state) >> np.uint32(taps[n][2])
        s ^= np.uint32(state) >> np.uint32(taps[n][3])

    s = (np.uint32(state) >> np.uint32(1)) | (np.uint32(s) << np.uint32(n))
    return np.uint32(s)

# class Tanh_Custom(BinaryScalarOp):
#     def impl(self, x, n):
#         # x_val, n = x
#         buckets = np.linspace(-3, 3, n)
#         idx = np.abs(buckets - x_val).argmin()
#         return np.tanh(np.take(buckets, idx))

# tanh_approx_scalar = Tanh_Custom(upgrade_to_float, name='tanh_var')
# tanh_var = Elemwise(tanh_approx_scalar)


global x_idx, idx, val
# x_idx = np.linspace(-3, 3, pow(2,nbits))
# idx = np.arange(0, pow(2,nbits))
# val = np.tanh(x_idx)#TODO

def tanh_pwl(x):
    # ret = T.switch(T.le(x,-1), -1, x)
    # ret = T.switch(T.lt(x, -0.3333) & T.gt(x, -1), 0.5*x-0.5, ret)
    # ret = T.switch(T.ge(x, -0.3333) & T.le(x, 0.3333), 2*x, ret)
    # ret = T.switch(T.gt(x, 0.3333) & T.lt(x, 1), 0.5*x+0.5, ret)
    # ret = T.switch(T.ge(x,1), 1, ret)

    ret = T.switch(
        T.ge(x,1), 1, 
        T.switch(
            T.gt(x, 0.4286) & T.lt(x, 1), 0.25*x+0.75, 
            T.switch(
                T.ge(x, -0.4286) & T.le(x, 0.4286), 2*x, 
                T.switch(
                    T.lt(x, -0.4286) & T.gt(x, -1), 0.25*x-0.75, 
                    T.switch(T.le(x,-1), -1, x)
    ))))
    return ret

def get_indices(x_idx, ret, x):
    # x = T.printing.Print('x')(x)
    # x_idx = T.printing.Print('x_idx')(x_idx)
    # ret = T.printing.Print('r_old')(ret)
    # ret = T.switch(x_idx <= x, x_idx, ret)
    # ret = T.printing.Print('r_new')(ret)
    # return ret
    return T.switch(x_idx <= x, x_idx, ret)

# def tanh_approx(x, nbits, x_idx):
def tanh_approx(x, x_idx):
    # ret = -3*T.ones_like(x, dtype=theano.config.floatX)
    ret = T.zeros_like(x, dtype=theano.config.floatX)
    ret, updates = theano.scan(
        fn=get_indices, 
        # outputs_info=-3*T.ones_like(x, dtype=theano.config.floatX), 
        outputs_info=T.zeros_like(x, dtype=theano.config.floatX), 
        sequences=[x_idx],
        non_sequences=[x] )
        # n_steps=nbits

    # ret = T.printing.Print('res')(ret)
    ret = ret[-1]
    # ret = T.printing.Print('res')(ret)
    return T.tanh(ret)
    

#printable np version
# def tanh_approx(x, nbits, x_idx):
#     ret = -3*np.ones(x.shape)
#     for i in range(2**nbits):
#         for idx, r in np.ndenumerate(ret):
#             ret[idx] = x_idx[i] if x_idx[i] <= x[idx] else r
#     print(ret)
#     return np.tanh(ret)

# def tanh_approx(x, n):
#     n = 2**n
#     buckets = np.linspace(-3, 3, n)
#     x_idx = x*1
#     x_idx = np.abs(buckets - x).argmin()
#     # for idx, x_val in np.ndenumerate(x):
#     #     x_idx[idx] = np.abs(buckets - x_val).argmin()

#     return np.tanh(np.take(buckets, x_idx))

# tanh_var = Elemwise(tanh_approx)


def nbitrandint(n, shape):
    return np.random.randint(low=1, high=pow(2,n), size=shape, dtype=np.uint16)
	
#discrete the delta_W from real value to be k*L, where k is an integer and L is the length of state step, i.e. 2H/(2^N)
def discrete_grads(loss,network,LR):
    global update_type,best_params,H,N,th, states, nbits_lfsr, nbits_tanh # th is a parameter that controls the nonlinearity of state transfer probability
    global tanh_type, rng_type

    W_params = lasagne.layers.get_all_params(network, discrete=True) #Get all the weight parameters
    layers = lasagne.layers.get_all_layers(network)
	
    W_grads = []
    for layer in layers:
        params = layer.get_params(discrete=True)
        if params:
            W_grads.append(theano.grad(loss, wrt=layer.W)) #Here layer.W = weight_tune(param) 
    updates = lasagne.updates.adam(loss_or_grads=W_grads,params=W_params,learning_rate=LR)  
    new_states = []
	
    x_idx = np.linspace(0, np.sqrt(1), nbits_tanh+1, dtype=theano.config.floatX)
    x_idx *= x_idx
    for param, parambest, state in izip(W_params, best_params, states) :

        L = 2*H/pow(2,N) #state step length in Z_N 
		
        a=random.random() #c is a random variable with binary value
        if a<0.8:
           c = 1
        else:
           c = 0
       
        b=random.random()
        state_rand = T.round(b*pow(2,N))*L - H #state_rand is a random state in the discrete weight space Z_N
       
        delta_W1 =c*(state_rand-parambest) #parambest would transfer to state_rand with probability of a, or keep unmoved with probability of 1-a
        delta_W1_direction = T.cast(T.sgn(delta_W1),theano.config.floatX)
    	dis1=T.abs_(delta_W1) #the absolute distance
        k1=delta_W1_direction*T.floor(dis1/L) #the integer part
        v1=delta_W1-k1*L #the decimal part
        Prob1= T.abs_(v1/L) #the transfer probability
        # Prob1 = T.tanh(th*Prob1) #the nonlinear tanh() function accelerates the state transfer
        Prob1 = T.cast(Prob1, theano.config.floatX)
        if tanh_type == "tanh":
            Prob1 = T.tanh(th*Prob1)
        elif tanh_type == "pwl":
            Prob1 = tanh_pwl(Prob1)
        elif tanh_type == "lut":
            Prob1 = tanh_approx(th*Prob1, th*x_idx)
        # Prob1 = tanh_var(th*Prob1, 2**nbits_tanh)
		   
        delta_W2 = updates[param] - param 
        delta_W2_direction = T.cast(T.sgn(delta_W2),theano.config.floatX)	   
        dis2=T.abs_(delta_W2) #the absolute distance
        k2=delta_W2_direction*T.floor(dis2/L) #the integer part
        v2=delta_W2-k2*L #the decimal part
        Prob2= T.abs_(v2/L) #the transfer probability
        # Prob2 = T.tanh(th*Prob2) #the nonlinear tanh() function accelerates the state transfer
        Prob2 = T.cast(Prob2, theano.config.floatX)
        if tanh_type == "tanh":
            Prob2 = T.tanh(th*Prob2)
        elif tanh_type == "pwl":
            Prob2 = tanh_pwl(Prob2)
        elif tanh_type == "lut":
            Prob2 = tanh_approx(th*Prob2, th*x_idx)
        # Prob2 = tanh_approx(th*Prob2, x_idx)
        # Prob2 = tanh_var(th*Prob2, 2**nbits_tanh)
        # Prob2 = tanh_pwl(Prob2)
        
        srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        Gate1 = T.cast(srng.binomial(n=1, p=Prob1, size=T.shape(Prob1)), theano.config.floatX) # Gate1 is a binary variable with probability of Prob1 to be 1
        if rng_type == "srng":
            Gate2 = T.cast(srng.binomial(n=1, p=Prob2, size=T.shape(Prob2)), theano.config.floatX) # Gate2 is a binary variable with probability of Prob2 to be 1
        else:
            state = lfsr_n(state, nbits_lfsr)
            new_states.append(state)
            Gate2 = T.switch(T.lt(state, Prob2*pow(2,nbits_lfsr)), 1, 0)

        delta_W1_new=(k1+delta_W1_direction*Gate1)*L #delta_W1_new = k*L where k is an integer   
        updates_param1 = T.clip(parambest + delta_W1_new,-H,H)
        # updates_param1 = T.clip(state_rand,-H,H)
        # updates_param1 = state_rand
        updates_param1 = weight_tune(updates_param1,-H,H) #fine tuning for guaranteeing each element strictly constrained in the discrete space

        delta_W2_new=(k2+delta_W2_direction*Gate2)*L #delta_W2_new = k*L where k is an integer  
        updates_param2 = T.clip(param + delta_W2_new,-H,H)
        updates_param2 = weight_tune(updates_param2,-H,H) #fine tuning for guaranteeing each element strictly constrained in the discrete space

	# if update_type<100, the weight probabilistically tranfers from parambest to state_rand, which helps to search the global minimum
        # elst it would probabilistically transfer from param to a state nearest to updates[param]		
        updates[param]= T.switch(T.lt(update_type,100), updates_param1, updates_param2)    

    states = new_states
    return updates


def train(  network,
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test):    
    
    global update_type,best_params,H,N,th, states, nbits_lfsr, nbits_tanh
    global lr_start, lr_fin, tanh_type, rng_type, seed, bn
    # A function which shuffles a dataset
    def shuffle(X,y):
    
        shuffled_range = range(len(X))
        np.random.shuffle(shuffled_range)
        
        new_X = np.copy(X)
        new_y = np.copy(y)
        
        for i in range(len(X)):           
            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]
            
        return new_X,new_y
    
    #train the network for one epoch on the training set 
    def train_epoch(X,y,LR):    
        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            new_loss = train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
            loss += new_loss
            print("Batch " +str(i), end='\r')
            sys.stdout.flush()
            
        
        loss/=batches

        return loss
    
    # Test the network on the validation set
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100 
        loss /= batches

        return err, loss               
    
    # shuffle the training set
    X_train,y_train = shuffle(X_train,y_train)
	# initialize the err to be 100%
    best_val_err = 100
    best_test_err = 100
	
	#initialize the best parameters
    best_epoch = 1
    best_params = lasagne.layers.get_all_params(network, discrete=True)
    update_type = 200 #intialize the update_type to be normal training
	
    verr = []
    tloss = []
    
    for epoch in range(num_epochs): 
        
		# if a new round of training did not search a better result for a long time, the network will transfer to a random state and continue to search
		# otherwise, the network will be normally trained
        if  epoch >= best_epoch + 15:
	        update_type = 10       
        else:
            update_type = 200 
        
        if epoch==0: # epoch 0 is for weight initialization to a discrete space Z_N without update
            LR = 0
            # _ = train_fn(X_train[0:10],y_train[0:10],LR)
            # best_params = lasagne.layers.get_all_params(network, discrete=True)
            # print("Initialized weights to discrete space.")
            # continue
        elif epoch<=1:
            LR = LR_start

        else:
            LR = LR*LR_decay #decay the LR  

        start_time = time.time()


        train_loss = train_epoch(X_train,y_train,LR)
        
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)
        test_err, test_loss = val_epoch(X_test,y_test)
		
        if epoch>=1: #collect data for plot
            tloss.append(train_loss)
            verr.append(val_err)
	       
        if test_err <= best_test_err:            
            best_test_err = test_err
            best_epoch = epoch + 1
            best_params = lasagne.layers.get_all_params(network, discrete=True)
	
        epoch_duration = time.time() - start_time

        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  update_type:                   "+str(update_type)) 
        print("  LR:                            "+str(LR))
        print("  th:                            "+str(th))
        print("  LR_decay:                      "+str(LR_decay))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best test error rate:          "+str(best_test_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        
     
    path = 'H'+str(H)+'N'+str(N)+'LR'+str(LR_start)+'D'+str(LR_decay)+'B'+str(batch_size)+'E'+str(num_epochs)+'tanh'+str(th)+'.mat'
    scio.savemat(path,{'valid_err':verr,'train_loss':tloss})
    
    fig = plt.figure(1) 
    x = np.arange(num_epochs-1) + 1
    sub1 = fig.add_subplot(211) 
    line1 = sub1.plot(x,verr,'r-',linewidth=2) 
    plt.xlabel('training epoch')
    plt.ylabel('validation error rate')
    sub2 = fig.add_subplot(212)
    line2 = sub2.plot(x,tloss,'b-',linewidth=2) 
    plt.xlabel('training epoch')
    plt.ylabel('training_loss')
    
    # plt.show()
    fig.savefig('out_'+str(lr_start)+'_'+str(lr_fin)+'.png')
    #print("###############nbits = "+str(nbits))
    with open('tanh_results.txt', 'a') as file:
        file.write(str(nbits_lfsr)+" "+str(nbits_tanh)+" "+str(best_test_err)+" "+str(best_epoch)+"\n")
    with open("verr_loss_"+tanh_type+str(nbits_tanh)+"_"+rng_type+str(nbits_lfsr)+"_"+str(seed)+".txt", 'a') as file:
        file.write("lr_start="+str(lr_start)+
            " lr_fin="+str(lr_fin)+
            " batchsize="+str(batch_size)+
            " num_epochs="+str(num_epochs)+
            " tanh="+tanh_type+
            " rng="+rng_type+
            " batchnorm="+str(bn)+"\n")
        file.write(str(best_test_err)+" "+str(best_epoch)+"\n")
        for ep, v, l in izip(x, verr, tloss):
            file.write(str(ep)+" "+str(v)+" "+str(l)+"\n")
    return best_test_err

    
    
def main(num_epochs, discrete, batch_size, lfsr_bits=7, tanh_bits=5, 
    start=0.1, fin=0.000001, tanh_t="tanh", rng_t="srng", seed_val=123, batchnorm=False):
	# BN parameters
    alpha = 0.1 
    print("alpha = "+str(alpha))
    epsilon = 1e-4 
    print("epsilon = "+str(epsilon))
	
    batch_size = batch_size 
    #batch_size = 64#TODO
    print("batch_size = "+str(batch_size))
       
    # Training parameters
    num_epochs = num_epochs
    print("num_epochs = "+str(num_epochs))
    

    activation = discrete_neuron_3states #activation discretization
    print("activation = discrete_neuron_3states")
	
    discrete = discrete
    #discrete = False
    print("discrete = "+str(discrete))
    
    global update_type,best_params,H,N,th, states, nbits_lfsr, nbits_tanh
    global lr_start, lr_fin, tanh_type, rng_type, seed, bn
    tanh_type = tanh_t
    rng_type = rng_t
    seed = seed_val
    bn = batchnorm

    H = 1. # the weight is in [-H, H]
    print("H = "+str(H))
    N = 1. # the state number of the discrete weight space is 2^N+1
    print("N = "+str(N)+" Num_States = "+str(pow(2,N)+1))
    th = 3.   #the nonlinearity parameter of state transfer probability
    print("tanh = "+str(th))

    
    # Decaying LR 
    LR_start = 0.1 
    LR_start = start
    lr_start = start
    print("LR_start = "+str(LR_start))
    LR_fin = 0.04 
    LR_fin = fin
    lr_fin = fin
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./(num_epochs)) 
    print("LR_decay = "+str(LR_decay))

    nbits_lfsr = lfsr_bits
    nbits_tanh = tanh_bits


    print('Loading MNIST dataset...')
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    test_set = MNIST(which_set= 'test', center = False)
    
    train_set.X = 2*train_set.X.reshape(-1, 1, 28, 28)-1.
    valid_set.X = 2*valid_set.X.reshape(-1, 1, 28, 28)-1.
    test_set.X = 2*test_set.X.reshape(-1, 1, 28, 28)-1.

    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.



    # x = np.linspace(-1.5, 1.5, 1024, dtype='float32')
    # inp = T.vector('tanh_x')
    # idx = T.vector('tanh_idxs')
    # nb = T.scalar('num_bits')
    # t_approx = tanh_approx(th*inp, th*idx)
    # tanh_fn = theano.function([inp, idx], t_approx)
    # fig,ax = plt.subplots()
    # ax.plot(x, np.tanh(th*x), label='tanh', color='black')
    # for i, c in izip([4, 8, 16, 32], ['tab:blue', 'tab:green', 'tab:red']):
    #     x_idx = np.linspace(0, 1, i+1, dtype=theano.config.floatX)
    #     print(x_idx)
    #     approx = tanh_fn(x, x_idx)
    #     ax.plot(x, approx, label=str(i), color=c)

    # ax.set(xlabel='x', ylabel='y')
    # legend = ax.legend(shadow=True, fontsize='x-large')
    # plt.show()
    # fig.savefig('tanh_function_approx.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    # return

    # x = np.linspace(-1.5, 1.5, 1024, dtype='float32')
    # inp = T.vector('tanh_x')
    # t_approx = tanh_pwl(inp)
    # tanh_fn = theano.function([inp], t_approx)
    # fig,ax = plt.subplots()
    # ax.plot(x, np.tanh(3*x), label='tanh', color='black')
    # approx = tanh_fn(x)
    # ax.plot(x, approx, label='pwl', color='tab:green')

    # ax.set(xlabel='x', ylabel='y')
    # legend = ax.legend(shadow=True, fontsize='x-large')
    # plt.show()
    # return

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')

    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    update_type = 200 #intialize the update_type to be normal training

    cnn = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input) 
    
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
            N=N,
            num_filters=32, 
            filter_size=(5, 5),
            pad = 'valid',
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
	
    if batchnorm:
        cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
				
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
			
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
            N=N,
            num_filters=64, 
            filter_size=(5, 5),
            pad = 'valid',
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
	
    if batchnorm:
        cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
				
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    cnn = DenseLayer(
                cnn, 
                discrete=discrete,
                H=H,
                N=N,
                num_units=512,
                nonlinearity=lasagne.nonlinearities.identity) 
    
    if batchnorm:
        cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
 
    cnn = DenseLayer(
                cnn, 
                discrete=discrete,
                H=H,
                N=N,
                num_units=10,
                nonlinearity=lasagne.nonlinearities.identity) 
    if batchnorm:
        cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    best_params = lasagne.layers.get_all_params(cnn, discrete=True)

    # lfsr init
    p_v = lasagne.layers.get_all_param_values(cnn, discrete=True)
    states = []
    np.random.seed(seed)
    for p in p_v:
        # state = np.array(np.random.uniform(1, 255, state.shape), np.uint8)
        state = np.random.randint(low=1, high=2**nbits_lfsr, size=p.shape, dtype=np.uint32)
        states.append(state)
	# print(states[0][0])

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))

    if discrete:  
        updates = discrete_grads(loss,cnn,LR)
        params = lasagne.layers.get_all_params(cnn, trainable=True, discrete=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
	def fix_update_bcasts(updates):
	    for param, update in updates.items():
		if param.broadcastable != update.broadcastable:
		    updates[param] = T.patternbroadcast(update, param.broadcastable)
	fix_update_bcasts(updates)
        
    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)


    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    train_fn = theano.function([input, target, LR], loss, updates=updates)
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    train(  cnn,
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y)


if __name__ == "__main__":
    num_lfsr = 0
    num_tanh = 0
    num_epochs = 0
    batch_size = 0
    discrete = False
    tanh = "tanh"
    rng = "srng"
    lr_start = 0.1
    lr_fin = 0.000001
    seed = 123

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='num_epochs', type=int)
    parser.add_argument('-d', dest='discrete', action='store_true')
    parser.add_argument('-b', dest='batch_size', type=int)
    parser.add_argument('--nl', dest='num_lfsr', type=int)
    parser.add_argument('--nt', dest='num_tanh', type=int)
    parser.add_argument('--lrs', dest='lr_start', type=float)
    parser.add_argument('--lrf', dest='lr_fin', type=float)
    parser.add_argument('-t', dest='tanh')
    parser.add_argument('-r', dest='rng')
    parser.add_argument('-s', dest='seed', type=int)
    parser.add_argument('--bn', dest='batchnorm', action='store_true')


    args = parser.parse_args()


    # try:
    #     opts, _ = getopt.getopt(sys.argv[1:], "he:b:dt:r:s:", ["nl=", "nt=", "lrs=", "lrf="])
    #     print(opts)
    # except getopt.GetoptError:
    #     print("commandline args error")
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print('python lfsr_mnist_GXNOR.py \
    #                 -nl [lfsr bits] \
    #                 -nt [tanh size] \
    #                 -e [num epochs] \
    #                 -d [discrete] \
    #                 -b [batchsize]\
    #                 -t [tanh]\
    #                 -r [rng]')
    #         sys.exit()
    #     elif opt in ('--nl'):
    #         num_lfsr = int(arg)
    #     elif opt in ('--nt'):
    #         num_tanh = int(arg)
    #     elif opt in ('-e'):
    #         num_epochs = int(arg)
    #         print("\nEPOCHS = "+str(num_epochs)+"\n")
    #     elif opt in ('-b'):
    #         batch_size = int(arg)
    #         print("\nBATCHSIZE = "+str(batch_size)+"\n")
    #     elif opt in ('-d'):
    #         discrete = True
    #     elif opt in ('-t'):
    #         tanh = str(arg)
    #     elif opt in ('-r'):
    #         rng = str(arg)
    #     elif opt in ('--lrs'):
    #         lr_start = int(arg)
    #     elif opt in ('--lrf'):
    #         lr_fin = int(arg)
    #     elif opt in ('-s'):
    #         seed = int(arg)
    main(args.num_epochs, args.discrete, args.batch_size, args.num_lfsr, args.num_tanh, args.lr_start, args.lr_fin, args.tanh, args.rng, args.seed, args.batchnorm)
