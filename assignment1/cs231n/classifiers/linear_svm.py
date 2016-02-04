import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    #print "dW's shape", dW.shape
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for train_image in xrange(num_train):
        #print "Train Image: ",train_image
        scores = X[train_image].dot(W)
        #print "Xi = ", X[train_image]
        #print "Scores: ",scores
        correct_class_score = scores[y[train_image]]
        for classifier in xrange(num_classes):
            if classifier == y[train_image]:
                continue
            margin = scores[classifier] - correct_class_score + 1  # note delta = 1
            #print "Margin: ", margin
            #print "Margin Shape:", margin.shape
            # And compute the gradient
            if margin > 0:
                loss += margin
                #Gradient
                #print "Classifier" ,classifier
                derivative = np.transpose(X[train_image])
                dW[:,y[train_image]] -= derivative
                dW[:,classifier] += derivative


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    # Add regularization gradient
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # Setting up
    dimension = X.shape[1]
    train_images = X.shape[0]
    classifier_classes = W.shape[1]
    # print "Number of train images",train_images
    # print "Number of classifier classes: ", classifier_classes
    all_scores = X.dot(W)

    # Those are the debugging statements that helped me solve this problem
    # print("Dimension: ",dimension)
    # print("X's shape"), X.shape
    #print "Shape of the scores", all_scores.shape
    #print "Y's shape",y.shape


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    # Determine all the scores for Score(correc_label) for all the 500 images

    # Correct scores should be a 500x1 vector, where each entry corresponds to one image' correct score
    correct_label_scores = all_scores[np.arange(train_images),y]

    #DEBUG
    # print "Correct scores shape, should be a 500x1 vector, where each entry corresponds to one image' correct score " \
    #       "scores", correct_label_scores.shape
    # print "Before subtracting", np.transpose(all_scores)

    # Subtracting the correct scores from all classifiers' scores
    margin = np.transpose(all_scores) - correct_label_scores + 1
    margin[y,np.arange(train_images)] = 0 # Subtracting the correct class examples

    #DEBUG
    # print "After", margin
    # margin = np.transpose(margin)
    # print "The margin's shape", margin.shape
    # print ""
    # print "The margin itself", margin

    #Computing the maximum
    zero_matrix = np.zeros((margin.shape))
    max_values = np.maximum(zero_matrix,margin )

    # Now, compute the loss for all examples
    loss = np.sum(max_values)
    loss /= train_images
    # And regularize everything :D
    loss += 0.5*reg*np.sum(W*W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Determine which classifiers have a loss > 0
    non_zero_classifiers = max_values
    non_zero_classifiers[max_values > 0] = 1
    # Calculating indexes for the necessary subtractions
    images_sum = np.sum(non_zero_classifiers, axis=0)
    # Subtracting the derivative
    non_zero_classifiers[y, range(train_images)] = -images_sum[range(train_images)]
    # And updating the gradients
    dW = np.transpose(np.dot(non_zero_classifiers, X))

    # Normalizer, normalizer, oh! He's a normalizer baby!
    dW /= train_images

    # Finally, regularize
    dW += reg*W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    # Yaaaaay!
    return loss, dW
