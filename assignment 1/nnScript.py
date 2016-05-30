import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pdb
import pickle
import time

t0 = time.time()
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    data_mat = loadmat('mnist_all.mat')#loads the MAT object as a Dictionary
    #Pick a reasonable size for validation data
    
    train_data=np.empty((0,784))
    test_data=np.empty((0,784))
    train_label=np.empty((0,1))
    test_label=np.empty((0,1))
	
    for i in range(10):
        train_data_temp = data_mat.get('train'+str(i))
        test_data_temp = data_mat.get('test'+str(i))
        train_data = np.vstack([train_data,train_data_temp])
        test_data = np.vstack([test_data,test_data_temp])
        for j in range(train_data_temp.shape[0]):
            train_label = np.append(train_label,i)
        for j in range(test_data_temp.shape[0]):
            test_label = np.append(test_label,i)
    
	
    #Randomize data and split
    # Train data size : 50000
    # Validation data size: 10000
    range_train_data = range(train_data.shape[0])
    perm = np.random.permutation(range_train_data)
    train_data_temp = train_data[perm[0:50000],:]
    validation_data_temp = train_data[perm[50000:],:]
    train_label_temp = train_label[perm[0:50000]]
    validation_label_temp = train_label[perm[50000:]]

	
    #Feature selection, threshold = 0.75
    features = np.invert(np.all(train_data - train_data[0,:] < 3, axis=0))
    train_data_temp = train_data_temp[:,features]
    validation_data_temp = validation_data_temp[:,features]
    test_data = test_data[:,features]

    #Normalization
    train_data_temp = np.divide(train_data_temp,255.0)
    test_data = np.divide(test_data,255.0)
    validation_data_temp = np.divide(validation_data_temp,255.0)
	
    #pdb.set_trace()
		
    train_data = train_data_temp
    validation_data = validation_data_temp
    train_label = train_label_temp
    validation_label = validation_label_temp
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    

    
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    y=np.zeros((training_label.shape[0],10))    
    for i in range (training_label.shape[0]):    
        y[i][training_label[i]]=1

    training_data = np.append(training_data,np.zeros([len(training_data),1]),1)
    n=training_data.shape[0]
    
    #Feedforward
    p = np.dot(training_data,w1.T)
    z = sigmoid(p)
    z = np.append(z,np.zeros([len(z),1]),1)
    q = np.dot(z,w2.T)        
    o = sigmoid(q)    
        
    delta = (y-o)*(1-o)*o
    
	# calculate error between true label and predicted label.
    error1 = ((y - o)**2)/2.0
    
	# Back Propogate error to recalculate weights.
    dot1 = np.dot(delta,w2)
	
    g2 = -1*(np.dot(delta.T,z))
    
    dot1 = dot1*z*(z - 1)
    
    g1 = np.dot(training_data.T,dot1)
    
	
    g1 = np.delete(g1, (n_hidden), axis=1)
    
	# Regularization using lambdaval
	
    fg1=(g1.T+(lambdaval*w1))/n
    
    fg2=(g2+(lambdaval*w2))/n

    obj_grad = np.concatenate((fg1.flatten(), fg2.flatten()),0)
    
    obj_val= (np.sum(error1)/n) + ((lambdaval/(2 * n))*(np.sum(np.square(w1)) + np.sum(np.square(w2))))
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    data = np.append(data,np.zeros([len(data),1]),1)
	
	# add bias input node
    data_size=data.shape[0]    
    
    #Feedforward
	
	# product of weights from input to hidden layer and input layer.
    p = np.dot(data,w1.T)
	
	# sigmoid of input layer
    
    z = sigmoid(p)
	
    # add bias hidden node
    
    z = np.append(z,np.zeros([len(z),1]),1)
	
    # product of weights from hidden to output layer and output of hidden layer.
    
    q = np.dot(z,w2.T)
    
    # sigmoid of output layer	
    
    o = sigmoid(q)    
    
    index = np.argmax(o,axis=1)
    
    # predict digits based on maximum value in the output layer.
    pred = np.empty((0,1))
    
    for i in range(data_size):
        index = np.argmax(o[i])
        pred = np.append(pred,index)
    
    return pred
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.6;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

t1 = time.time()
print ('\n Time taken: ' + str(t1 - t0) + ' seconds, with lambdaval: ' + str(lambdaval) + ' and hidden nodes: ' + str(n_hidden))

pickle.dump( [n_hidden, w1, w2, lambdaval], open( "params.pickle", "wb" ))