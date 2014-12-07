import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import minimize



class LinearRegression:
    """LinearRegression class - one-step real-value linear regression for 
        binary classification based on Caltech's machine learning class Lecture
        03 by Yaser Abu-Mostafa
        Class instance variables:
            phi - transformation function for input space X, default to linear
            Z - transformed space from X, 2-D numpy array of Zn
            X - input space X, 2-D numpy array of Xn
            Y - binary +/-1 classification results for input X, 1-D numpy array
                of +/-1
            g - binary linear regression best hypothesis results for input X,
                1-D numpy array of +/-1
            w - linear regression weighting factors corresponding to zn, 1-D 
                numpy array
            inSampleError - average in-sample error using 0 or 1 binary error
                sum between Y and g
        Methods:
            update_regression() - compute linear regression w and g based on 
                the training data set
            update_model_results() - compute the model results g and the error
                function for the given weight vector
            plot_compare_data_2d() - plot side-by-side scatter plots of 
                training results vs. linear regression classification results
    """

    def __init__(self, trainingData, transformFunctionName = 'linear'):
        # determine the input X transformation function
        if transformFunctionName.lower() == 'linear':
            phi = phi_linear
        elif transformFunctionName.lower() == 'quadratic':
            phi = phi_quadratic
        
        else:   # default linear function
            phi = phi_linear
        
        self.phi = phi
        
        # transform the input X to Z; w only exists in Z
        Z = []
        for n in range(len(trainingData.X)):
            Z.append(self.phi(trainingData.X[n]))
        self.Z = np.array(Z)
        self.X = trainingData.X
        self.Y = trainingData.Y
        self.g = np.ones(len(self.Y))
        self.w = np.ones(len(Z[0]))
        self.inSampleError = 0

    def update_linear_regression(self):
        # transpose and inverse function only works for matrix
        # compute one-step linear regression pseudo inverse
        Z = np.matrix(self.Z)
        ZT = np.transpose(Z)
        try:
            invZTZ = inv(ZT*Z)
        except:
            return
        pseudoInvZ =invZTZ*ZT
        
        # transform Y into a single column matrix
        YT = np.transpose(np.matrix(self.Y))
        w = pseudoInvZ*YT
        self.w = np.array(np.transpose(w))[0]
        self.update_model_results()

    def update_model_results(self):
        # compute the guessed hypothesis result g based on the computed w
        g = []
        for n in range(len(self.Z)):
            g.append(np.dot(self.Z[n], self.w))
        self.g = np.array(np.sign(g))
        
        # compute the error function using average of binary sum
        e = self.g - self.Y
        # need floating conversion to get result without int rounding
        self.inSampleError = float(len(e[e != 0]))/float(len(self.g))    



    def plot_compare_data_2d(self):
        # create two subplots side-by-side with the same y axis
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.suptitle('Learned Results Validation')
        errFcnString = 'In-sample error: ' + str(100*self.inSampleError) + '%'
        fig.text(0.4, 0.02, errFcnString)
        plot_scatter_2d(self.X, self.Y, ax1, 'Training Data', 'x1', 'x2', \
            [0, 1.0], [0, 1.0])
        plot_scatter_2d(self.X, self.g, ax2, 'Learned Model', 'x1', 'x2', \
            [0, 1.0], [0, 1.0])
        plt.show()


class Perceptron(LinearRegression):
    """Perceptron class - percetron linear model for binary classification 
        based on Caltech's machine learning class Lecture 01 by Yaser 
        Abu-Mostafa
        Inherit from LinearRegression class with the same initilization
            parameters
        Class instance variables:
            phi - transformation function for input space X, default to linear
            Z - transformed space from X, 2-D numpy array of Zn
            X - input space X, 2-D numpy array of Xn
            Y - binary +/-1 classification results for input X, 1-D numpy array
                of +/-1
            g - binary linear regression best hypothesis results for input X,
                1-D numpy array of +/-1
            w - linear regression weighting factors corresponding to zn, 1-D
                numpy array
            inSampleError - average in-sample error using 0 or 1 binary error
                sum between Y and g
        Methods:
            update_perceptron() - compute perceptron w and g based on
                the training data set within the maximum iterations
    """
    
    def update_perceptron(self, maxIter = 1000):
        # initialize the weight with linear regression
        self.update_linear_regression()
        N = len(self.Y)     # number of training data points
        # iterate for the maximum number allowed
        for i in range(maxIter):
            mismatch = 0    # flag for mismatch
            for n in range(N):
                # classification result for the current weight vector
                hn = np.sign(np.dot(self.Z[n], self.w))
                if hn != self.Y[n]:
                    # weight vector adjustment when mismatched at nth point
                    self.w = self.w + (self.Y[n]*self.Z[n])
                    mismatch = 1
                    break
            if mismatch == 0:   # exit iteration when all matched
                break;
        # update the final g and error functions for the final weight vector
        self.update_model_results()       
                    
            
            
class LogisticsRegression(LinearRegression):
    """LogisticsRegression class - logistics regression linear model for binary
        classification based on Caltech's machine learning class Lecture 09 by
        Yaser Abu-Mostafa.  The logistics function is exp(s)/(1+exp(s)).
        Inherit from LinearRegression class with the same initilization
            parameters
        Class instance variables:
            phi - transformation function for input space X, default to linear
            Z - transformed space from X, 2-D numpy array of Zn
            X - input space X, 2-D numpy array of Xn
            Y - binary +/-1 classification results for input X, 1-D numpy array
                of +/-1
            g - binary linear regression best hypothesis results for input X,
                1-D numpy array of +/-1
            w - linear regression weighting factors corresponding to zn, 1-D
                numpy array
            inSampleError - average in-sample error using 0 or 1 binary error
                sum between Y and g
        Methods:
            update_logistics_regression() - compute perceptron w and g based on
                the training data set using Broyden-Fletcher-Goldfarb-Shanno
                algorithm
    """

    def update_logistics_regression(self):
        # initialize the weight with linear regression
        self.update_linear_regression()
        # iterate to obtain the optimum w using 
        #   Broyden-Fletcher-Goldfarb-Shanno algorithm
        res = minimize(logistics_ein, self.w, (self.Y, self.Z), \
            method = 'BFGS', jac = logistics_grad_ein, \
            options = {'disp':True})
        self.w = res.x
        # update the final g and error functions for the final weight vector
        self.update_model_results()

                                 
def logistics_ein(x, Y, Z):
    # compute the cross-entropy error function based on maximizing the
    #   likelihood
    # Input: x = weight factor w, Y = training data classification values, Nx1
    #   np array, Z = transformed training data set, NxdTilde np array
    # Output: ein = error function, scalar
    ZT = np.transpose(np.matrix(Z))    # dTilde x N matrix
    w = np.matrix(x)   # 1 x dTilde matrix
    s = np.array(w*ZT)[0]   # 1 x N array
    ein = np.mean(np.log(1 + np.exp(-Y*s)))
    return ein

def logistics_grad_ein(x, Y, Z):
    # compute the gradient of Ein error function
    # Input: x = weight factor w, Y = training data classification values, Nx1
    #   np array, Z = transformed training data set, NxdTilde np array
    # Output: gradEin = error function gradient, 1xN np array
    w = x
    N = len(Y)     # number of training data points
    gradEin = np.zeros_like(Z[0])  # initialized the grad Ein
    for n in range(N):
        # compute the prefactor for the sum over N
        kn = Y[n]/(1 + np.exp(Y[n]*np.dot(w, Z[n])))
        gradEin = gradEin + (kn*Z[n])
    gradEin = -gradEin/N
    return gradEin


class TrainingData:
    """TrainingData class - create input space X
        Class instance variables:
            X - training data input space, 2D numpy array with column dimension
                (d+1) and row number N for the total data points, each data
                point x[0] = 1 for the threshold w[0] setting
            Y - classification result vector corresponding to input X, 1D numpy
                array
        Methods:
            add_data(x, y) - add data points to the training set, x and y are
                the data list to be added
            generate_model_data(targetFcn, [N], [d], [distributionName]) - 
                generate a training data set based on the specified random
                distribution, targetFcn - target function, N - number of data
                points, d - dimension number for x (not includiing x[0]),
                distributionName - input space probability distribution 
                    function
            add_noise_uniform_binary(pYCondX) - add uniformly distributed noise
                to each training data point, i.e., the same distribution for 
                all X. pYCondX - probability of the Y result given X; for 
                binary noise, keep the same value if random number is not
                greater than pYCondX.
            plot_training_data_2d() - generate the scatter plot of the training
                data set
    """
    
    def __init__(self):
        self.X = np.array([])
        self.Y = np.array([])

    def add_data(self, x, y):
        if len(self.X) == 0:
            self.X = np.array([x])
            self.Y = np.array([y])
        else:
            self.X = np.array([self.X, x])
            self.Y = np.array([self.Y, y])

    def generate_model_data(self, targetFcn, N = 10, d = 2, distributionName \
        = 'uniform'):
        # generate X based on the probability distribution function in X
        X = []
        if distributionName.lower() == 'uniform':
            for n in range(N):
                xn = np.random.random_sample((d+1,))
                xn[0] = 1
                X.append(xn)
        else:   # default uniform distribution
            for n in range(N):
                xn = np.random.random_sample((d+1,))
                xn[0] = 1
                X.append(xn)
        # convert list to numpy array
        self.X = np.array(X)
        self.Y = targetFcn.eval_f_classified(self.X)

    def add_noise_uniform_binary(self, pYCondX):
        # uniform [0, 1] distribution for P(Y given X)
        p = np.random.random_sample((len(self.Y),))
        # only flip the binary result if the random number is above P(Y given X)
        self.Y[p > pYCondX] = -1*self.Y[p > pYCondX]


    def plot_training_data_2d(self):
        fig, ax = plt.subplots()
        fig.suptitle('Training Data Set')
        plot_scatter_2d(self.X, self.Y, ax, 'Training Data', 'x1', 'x2', \
            [0, 1.0], [0, 1.0])
        plt.show()



def plot_scatter_2d(X, Y, ax, title, xLabel, yLabel, xLim, yLim):
    # check to plot only for 2D results
    if len(X) > 0:
        if len(X[0]) == 3:
            # get x1 and x2 from the X array
            x1 = np.array([X[n][1] for n in range(len(X))])
            x2 = np.array([X[n][2] for n in range(len(X))])
            # get the (x1, x2) data point for the positive classification
            x1Pos = x1[Y > 0]
            x2Pos = x2[Y > 0]
            # get the (x1, x2) data point for the negative classification
            x1Neg = x1[Y < 0]
            x2Neg = x2[Y < 0]
            # blue and 'o' marker for +1
            ax.scatter(x1Pos, x2Pos, s=40, c='b', marker='o')
            # red and 'x' marker for -1
            ax.scatter(x1Neg, x2Neg, s=40, c='r', marker='x')
            ax.set_title(title)
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.set_xlim(xLim[0], xLim[1])
            ax.set_ylim(yLim[0], yLim[1])
        


class TargetTransformFunction:
    """TargetTrasnformFunction class - create model target function based on a
        given tranformation of X
        Class instance variables:
            name - target function name
            wTilde - weighting factors in the transformed space, 1D numpy array
            phi - input X tranformation function
        Methods:
            eval_f_tilde(X) - evalute the real value target function for the
                given input X, 1D numpy array
            eval_f_classified(X) - evaluate the binary classification results
                for the given input X, 1D numpy array
    """

    def __init__(self, name = 'linear', dTilde = 2, wTilde = []):
        self.name = name
        # if no input for wTilde, generate based on uniform distribution 
        #   in [-1, 1]
        if wTilde == []:
            # uniform random number from -1 to 1
            wTilde = 2*np.random.random_sample((dTilde+1,)) - 1
            # set the threshold to yield equal probability for both binary
            #   classification results
            # equal chance for +1 and -1 for an input range from 0 to 1
            wTilde[0] = -np.sum(wTilde[1:])/2
            self.wTilde = wTilde
        else:
            self.wTilde = np.array(wTilde)
        
        if name.lower() == 'linear':
            phi = phi_linear
        elif name.lower() == 'quadratic':
            phi = phi_quadratic        
        else:   # default linear function
            phi = phi_linear
        
        self.phi = phi

    def eval_f_tilde(self, X):
        fTilde = []
        for n in range(len(X)):
            z = self.phi(X[n])
            fTilde.append(np.dot(z, self.wTilde))
        fTilde = np.array(fTilde)
        return fTilde

    def eval_f_classified(self, X):
        Y = np.sign(self.eval_f_tilde(X))
        return Y



def phi_linear(x):
    # calculate the linear transform of x, a 1D numpy array
    z = x
    return z


def phi_quadratic(x):
    # calculate the quadratic transform of x, a 1D numpy array;
    #   expect nCr(n+k-1, k) elements, n = d, k =2
    d = len(x)
    # initialize z with the first threshold element to be the same as x0
    z = [x[0]]
    for i in range(1, d):       # cross multiply all combinations of xi and xj
        for j in range(i, d):
            z.append(x[i]*x[j])
    z = np.array(z)
    return z


