import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    # print "dW's shape", dW.shape
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax.ipynb loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # For every training image
    for train_image in xrange(num_train):
        # Multiply the weights by the image to get the scores
        scores = X[train_image].dot(W)
        # print(scores)
        # And then get the correct score
        correct_label = y[train_image]
        correct_score = scores[correct_label]
        # TODO: Right up to here
        #  And then get the score of every other classifier
        all_scores = np.sum(scores)
        # Add a normalizing factor for numeric stability
        normalizing_constant = np.max(scores)
        scores -= normalizing_constant
        correct_score -= normalizing_constant
        #Calculating the softmax values
        softmax = np.exp(correct_score)/np.sum(np.exp(scores))

        # print("Correct score softmax",softmax)

        # And calculating the loss
        loss += -1*np.log(softmax)
        # print loss
        #TODO: Loss computation is also correct

        # And calculating the gradient

        # First, update the Weight matrix with the correct example's derivative
        dW[:,correct_label] += (softmax-1)*np.transpose(X[train_image])

        # Then do the same for the wrong cases
        incorrect_labels = [x  for x in xrange(num_classes) if x != correct_label]
        # Now, update the weights
        for label_index in incorrect_labels:
          #Calculating the softmax for a wrong label
          incorrect_label_softmax = np.exp(scores[label_index])/(np.sum(np.exp(scores)))
          # Calculating the derivative
          necessary_weight = incorrect_label_softmax*np.transpose(X[train_image])
          # Updating the weights
          dW[:,label_index] += necessary_weight


    # Divide the loss
    loss /= num_train
    dW /= num_train

    # Now, do regularization
    loss += 0.5*reg*np.sum(W*W)# Penalize big weights
    dW += reg*W




    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

    #############################################################################
    # TODO: Compute the softmax.ipynb loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    train_images = X.shape[0]
    # Store all the scores in a matrix
    all_scores = np.dot(X,W)
    #First, calculate the normalizing constant for numeric stability
    constant = np.max(all_scores,axis=1)
    normalized_scores = np.transpose(np.subtract(np.transpose(all_scores),constant))

    #Then, calculate softmax for the correct scores
    exp_scores = np.exp(all_scores)
    # First, keep track of the sum of values per row
    exp_sum = np.sum(exp_scores,axis=1)

    # Finally, calculate the softmax score for every entry
    softmax_scores = np.transpose(exp_scores)/exp_sum # useful when computing gradient
    softmax_scores = np.transpose(softmax_scores)
    # And then, compute the loss
    loss_score = softmax_scores[range(train_images),y]
    loss_score = -1 * np.log(loss_score) #taking the logarithm
    loss += np.sum(loss_score)

    #Normalize and regularize the loss
    loss /= train_images
    loss += 0.5*reg*np.sum(W*W)

    #Finally, calculate a vectorized gradient

    # Calculate the derivative at the correct label
    softmax_scores[range(train_images),y] -= 1
    # Then, make a matrix containing all the gradient values
    gradient_values = np.dot(np.transpose(X),softmax_scores)
    gradient_values = gradient_values

    #FINALLY, update the gradient
    dW+= gradient_values
    #And normalize and regularize it
    dW /= train_images
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
