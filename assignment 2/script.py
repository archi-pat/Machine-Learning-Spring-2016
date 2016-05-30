#%matplotlib inline
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import pdb
import time

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    dim = X.shape[1] # dimension
    n_class = int(np.max(y))
    means = np.zeros((dim,n_class))

    covmat = np.zeros(dim)
    for i in range (0, n_class):
        idx = np.where(y==(i+1))[0]  # find all the rows of class i
        class_data = X[idx,:] 
        tmp_mean = np.mean(class_data, axis=0) # calculate the mean of data in the same class 
        means[:,i] = tmp_mean.transpose()
        
    covmat=np.cov(X,rowvar=0)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    dim = X.shape[1]
    n_class = int(np.max(y))    
    means = np.zeros((dim, n_class))
    covmats = [];
    for i in range (0, n_class):
        idx = np.where(y==(i+1))[0]
        class_data = X[idx,:] 
        tmp_mean = np.mean(class_data, axis=0) # calculate the mean of data in the same class 
        means[:,i] = tmp_mean.transpose()
        covmats.append(np.cov(class_data.transpose()))
    return means,covmats
	
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    n_class = means.shape[1]
    counter = 0.0
    inv_cov = np.linalg.inv(covmat)
    ypred = np.array([])
    for i in range (0, N):
        pdf = 0
        predict_class = 0
        tmp_mean = np.transpose(Xtest[i,:])
        for k in range (0, n_class):
            tmp1=np.dot(np.transpose(tmp_mean - means[:, k]), inv_cov)
            tmp2=tmp_mean - means[:, k]
            result = np.exp((-0.5)*np.dot(tmp1,tmp2))
            if (result > pdf):
                predict_class = k+1
                pdf = result
        if (ypred.shape[0]==0):
            ypred = np.array([[predict_class]])
        else:
            ypred = np.r_[ypred,[[predict_class]]]
        if (predict_class == ytest[i]):
            counter = counter + 1;
    acc = counter/N
    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    n_class = means.shape[1]
    normalizers = np.zeros(n_class)
    tmp_covmats = covmats
    dim = Xtest.shape[1]
    for i in range (0, n_class):
        normalizers[i] = 1.0/(np.power(2*np.pi, dim/2)*np.power(np.linalg.det(tmp_covmats[i]),0.5))
        tmp_covmats[i] = np.linalg.inv(tmp_covmats[i])
    N = Xtest.shape[0]
    counter = 0.0
    ypred = np.array([])
    for i in range (0, N):
        pdf = 0
        predict_class = 0
        testX = np.transpose(Xtest[i,:])
        for k in range (0, n_class):
            inv_cov = tmp_covmats[k]
            tmp1 = np.dot(np.transpose(testX - means[:, k]),inv_cov)
            tmp2 = testX - means[:, k]
            result = normalizers[k]*np.exp((-0.5)*np.dot(tmp1,tmp2))
            if (result > pdf):
                predict_class = k + 1
                pdf = result
        if (ypred.shape[0]==0):
            ypred = np.array([[predict_class]])
        else:
            ypred = np.r_[ypred,[[predict_class]]]
        if (predict_class == ytest[i]):
            counter = counter + 1
    acc = counter/N
    return acc, ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD    
    a = np.dot(X.T,X)
    b = np.dot(X.T,y)
    winv = np.linalg.inv(a)
    w = np.dot(winv, b) 
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD      
    w = np.linalg.inv((lambd * np.identity(X.shape[1])) + np.dot(X.T, X))
    w = np.dot(w, X.T)
    w = np.dot(w, y)
    return w


def squaredSum(w, X, y):
    wT = w.reshape((w.shape[0],1))
    return np.sum(np.square((y-np.dot(X,wT))))

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    rmse = np.sqrt((1.0/Xtest.shape[0]) * squaredSum(w, Xtest, ytest))
    return rmse

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD  
    # Error: 1/2*(y-w^Tx)^2 + 1/2*lambd*w^Tw
    #Squared Error : W^T*X^T*X - Y^T*X + lambd*W
    
    error = (0.5 * squaredSum(w, X, y)) + (0.5 * lambd * np.dot(w.T, w))
    error_grad = ((((np.dot(w.T, np.dot(X.T, X))))) +(-1.0 * np.dot(y.T, X)) + (lambd * w)).flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xd[:, i] = x ** i
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2

#start_time = time.time()
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

#print ("time", (time.time() - start_time))

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
#start_time = time.time()
k = 101
#lambdas = np.linspace(0, 1, num=k)
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
#rmses3_train = np.zeros((k,1))
#min_rmse3 = 9999.0
#min_lambda = 0
#min_weights = np.empty([X.shape[1], 1])

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    #rmses3_train[i] = testOLERegression(w_l,X_i,y)
    #if rmses3[i] < min_rmse3:
        #min_rmse3 = rmses3[i]
        #min_lambda = lambd
        #min_weights = w_l
    i = i + 1
#print ("time", (time.time() - start_time))
#print('Optimum lambda: ', min_lambda, ' at RMSE = ', min_rmse3)
#print('Variance (OLE): ', np.var(w_i), ', Variance (Ridge)', np.var(min_weights))

#Ridge Regression comparison
#plt.figure()
#plt.title("Problem 3: Ridge Regression")
#plt.plot(lambdas,rmses3)
#plt.plot(lambdas,rmses3_train)
#plt.xlabel('Lambda value')
#plt.ylabel('RMSE')
#plt.legend(('On test data','On training data'))
#plt.savefig('RMSE_Lambda.jpg')
#plt.show()

# Weight comparison
#plt.figure()
#plt.title("Weights for OLE Regression with Intercept")
#plt.plot(range(0, w_i.shape[0]),w_i)
#plt.xlabel('Weight value')
#plt.savefig('OLE_weights.jpg')
#plt.legend('OLE')
#plt.show()

#plt.figure()
#plt.title("Weights for Ridge Regression with Intercept")
#plt.plot(range(0, w_l.shape[0]),w_l)
#plt.xlabel('Weight value')
#plt.savefig('Ridge_Weights.jpg')
#plt.legend('RidgeRegression')
#plt.show()

plt.plot(lambdas,rmses3)	

# Problem 4
#start_time = time.time()
lambdas = np.linspace(0,1, num=k)
k = 101
i = 0
rmses4 = np.zeros((k,1))
#rmses4_train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    #w_l_1 = np.zeros((X_i.shape[1],1))
    #for j in range(len(w_l.x)):
        #w_l_1[j] = w_l.x[j]
    #rmses4_train[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1
#print(rmses4)
#print(rmses4_train)

#print ("time", (time.time() - start_time))
#plt.show()
#plt.plot(lambdas,rmses4,label='Test Data')
#plt.plot(lambdas,rmses4_train,label='Train Data')
#plt.xlabel('Lambda')
#plt.ylabel('RMSE')
#plt.title("Gradient Descent Ridge Regression-RMSE vs Lambda")
#plt.legend(loc='lower right',frameon=False)
#plt.savefig("Problem4.jpg")
#plt.clf()

plt.plot(lambdas,rmses4)

# Problem 5
#start_time = time.time()
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
#lambda_opt_train=lambdas[np.argmin(rmses4_train)]
#print(lambda_opt)
rmses5 = np.zeros((pmax,2))
#rmses5_train = np.zeros((pmax,2))
#min_p = 0
#min_rmse5 = 999999.9
#min_p_regu = 0
#min_rmse5_regu = 999999

p_opt1 = 1
p_opt2 = 2
N = X.shape[0]

#Xd_opt1 = np.empty([N, p_opt1 + 1])
#Xd_opt2 = np.empty([N, p_opt2 + 1])
#w_opt1 = np.empty([N, 1])
#w_opt2 = np.empty([N, 1])

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    #rmses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    #rmses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    #if p == p_opt1:
        #Xd_opt1 = Xdtest
        #w_opt1 = w_d1
    #if p == p_opt2:
        #Xd_opt2 = Xdtest
        #w_opt2 = w_d2
    
    #if rmses5[p,0] < min_rmse5:
        #min_rmse5 = rmses5[p,0]
        #min_p = p
      
    #if rmses5[p,1] < min_rmse5_regu:
        #min_rmse5_regu = rmses5[p,1]
        #min_p_regu = p  
    
#print ("time", (time.time() - start_time))    
#print ('Optimum p (no regu): ', min_p)
#print ('Optimum p (with regu): ', min_p_regu)

plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))

#plt.title('Non-linear Regression-RMSE vs P - Test Data')    
#plt.xlabel('P')
#plt.ylabel('RMSE')
#plt.plot(range(pmax),rmses5[:,0],label='No Regularization(lambda=0)')
#plt.plot(range(pmax),rmses5[:,1],label='Regularization(lambda='+str(lambda_opt)+')')
#plt.legend(loc='upper right',frameon=False)
#plt.savefig("Problem5.jpg")
#plt.clf()

#print (rmses5_train[:,0])
#print(rmses5_train[:,1])

#print rmses5 
#plt.title('Non-linear Regression-RMSE vs P - Train Data')    
#plt.xlabel('P')
#plt.ylabel('RMSE')
#plt.plot(range(pmax),rmses5_train[:,0],label='No Regularization(lambda=0)')
#plt.plot(range(pmax),rmses5_train[:,1],label='Regularization(lambda='+str(lambda_opt)+')')
#plt.legend(loc='upper right',frameon=False)
#plt.savefig("Problem5_1.jpg")
#print rmses5_train
#plt.clf()


#plt.figure()
#y_opt1 = np.dot(Xd_opt1, w_opt1)
#for i in range (Xtest.shape[0]):
    #plt.scatter(Xtest[i][2], ytest[i][0], c='b', marker='1')
#y_opt2 = np.dot(Xd_opt2, w_opt2)
#plt.title('Curve plots for non-linear regression')
#plt.plot(Xtest[:,2], y_opt1[:,0], c='r')
#plt.plot(Xtest[:,2], y_opt2[:,0], c='c')
#plt.legend(('No regularization', 'With regularization'))
#plt.savefig("Problem5_2.jpg")
#plt.show()