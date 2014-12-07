#!/usr/bin/python
import training_data_model as tdm


if __name__ == "__main__":
    """ Linear regression classification demo of 1 step learning of linear
        weight factors
            Input arguments:
                N = number of data points in the training set
                pYCondX = uniform binary noise addition according to the
                    probability distribution of Y given X
    """
    
    import sys
    N = int(sys.argv[1])        # number of training data points
    pYCondX = float(sys.argv[2])  # noise addition with P(Y given X)
    tf = tdm.TargetTransformFunction()      # linear d=2 target function
    
    # generate a set of N training data points uniformly distributed in X
    td = tdm.TrainingData()
	# N data points according to the target function
    td.generate_model_data(tf, N)  
    td.add_noise_uniform_binary(pYCondX)    # add noise

    # perform linear regression classification
    lr = tdm.LinearRegression(td)
    lr.update_linear_regression()
    # plot comparison result scatter charts
    lr.plot_compare_data_2d()
