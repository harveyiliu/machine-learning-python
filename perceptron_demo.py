#!/usr/bin/python
import training_data_model as tdm


if __name__ == "__main__":
    """ Linear perceptron classification demo of iterative linear perceptron
        weight factors optimization
            Input arguments:
                N = number of data points in the training set
                pYCondX = uniform binary noise addition according to the
                    probability distribution of Y given X
                maxIter = maximum number of iterations in updating the
                    perceptron weight vector
    """
    
    import sys
    N = int(sys.argv[1])        # number of training data points
    pYCondX = float(sys.argv[2])  # noise addition with P(Y given X)
    maxIter = int(sys.argv[3])  # maximum number of perceptron iterations
    tf = tdm.TargetTransformFunction()      # linear d=2 target function
    
    # generate a set of N training data points uniformly distributed in X
    td = tdm.TrainingData()
	# N data points according to the target function
    td.generate_model_data(tf, N)  
    td.add_noise_uniform_binary(pYCondX)    # add noise

    # perform perceptron classification
    pt = tdm.Perceptron(td)
    pt.update_perceptron(maxIter)
    # plot comparison result scatter charts
    pt.plot_compare_data_2d()
