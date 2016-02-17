import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(shape=hidden_dim)
        # Shape of W2 should be Hidden*numclasses
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(shape=num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
        scores = None
        w1,w2 = self.params['W1'], self.params['W2']
        b1,b2 = self.params['b1'], self.params['b2']
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Computing forward pass for the first layer
        layer1_relu,layer1_cache = affine_relu_forward(X,w1,b1)
        # Then, compute the forward pass for the second layer
        scores,layer2_cache = affine_forward(layer1_relu,w2,b2)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #Making a dictionary to store the gradients
        grads = {}

        # Finally, compute the loss and the scores
        loss, dx = softmax_loss(scores,y)


        # Backprop through the layers

        dlayer2,dw2,db2 = affine_backward(dout=dx, cache=layer2_cache)
        # First layer
        dlayer1,dw1,db1 = affine_relu_backward(dout = dlayer2,cache=layer1_cache)

        # # Some debugging
        # print "Softmax's shape", dx.shape
        # print "Second layer's cache's shape", layer2_cache[1].shape
        # print "Second affine output = ", dlayer2.shape
        # print "First layer's cache's shape", layer1_cache[1].shape
        # print "First affine and ReLU output", dlayer1.shape
        #

        print ""
        print "****"
        # Regularize the weights
        dw1 += self.reg*w1
        dw2 += self.reg*w2


        # And add the weight gradients to the grads dictionary
        grads['W1'] = dw1
        grads['W2'] = dw2
        grads['b1'] = db1
        grads['b2'] = db2

        # Finally, regularize the loss
        loss += 0.5 * self.reg * np.sum(w1*w1)
        loss += 0.5 * self.reg * np.sum(w2*w2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        print "Hidden dimensions", hidden_dims
        # Initializing the layers
        for layer in range(self.num_layers):

            # Making dictionary keys
            current_weight = "W"+str(layer+1)
            current_bias = "b"+str(layer+1)
            if use_batchnorm:
                current_gamma = "gamma"+str(layer+1)
                current_beta = "beta"+str(layer+1)
            print "Parameters",current_weight, current_bias


            # First Layer
            if layer == 0:

                print "Layer", layer," shape",(input_dim, hidden_dims[0])
                self.params[current_weight] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dims[0]))
                self.params[current_bias] = np.zeros(shape=hidden_dims[0])
                if use_batchnorm:
                    self.params[current_gamma] = np.ones(shape=hidden_dims[0])
                    self.params[current_beta] = np.zeros(shape=hidden_dims[0])
            # Last Layer
            elif layer == self.num_layers-1:
                print "Layer", layer," shape",(hidden_dims[-1],num_classes)
                self.params[current_weight] = np.random.normal(scale=weight_scale, size=(hidden_dims[-1],num_classes))
                self.params[current_bias] = np.zeros(shape = num_classes)
                if use_batchnorm:
                    self.params[current_gamma] = np.ones(shape=num_classes)
                    self.params[current_beta] = np.zeros(shape=num_classes)
            # Any other layer
            else:
                print "Layer", layer," shape",(hidden_dims[layer-1], hidden_dims[layer])
                self.params[current_weight]  = np.random.normal(scale=weight_scale, size=(hidden_dims[layer-1], hidden_dims[layer]))
                self.params[current_bias] = np.zeros(shape=hidden_dims[layer])
                if use_batchnorm:
                    self.params[current_gamma] = np.ones(shape=hidden_dims[layer])
                    self.params[current_beta] = np.zeros(shape=hidden_dims[layer])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # Stores the cached values for every layer
        caches = {}
        previous_weight = None
        previous_bias = None
        current_input = X

        for layer in range(self.num_layers-1):
            current_key = "W"+str(layer+1)
            current_bias = "b"+str(layer+1)
            layer_name = "Layer"+ str(layer+1)
            cache_dict = {}

            weights = self.params[current_key]
            bias = self.params[current_bias]

            # Storing this layer's cache
            affine_out,affine_cache = affine_forward(current_input,weights,bias)

            # If batch normalization was enabled
            if self.use_batchnorm:
                gamma = self.params["gamma"+str(layer+1)]
                beta = self.params["beta"+str(layer+1)]
                #TODO: Watch out for the indexing of BN_PARAMS
                batch_out, batch_cache =  batchnorm_forward(gamma=gamma,beta=beta,bn_param = self.bn_params[layer])
                #Store the cache
                cache_dict["batch"] = batch_cache

            # Now, do the forward pass for the ReLU layer
            relu_out,relu_cache = relu_forward(batch_out if self.use_batchnorm else affine_out)

            # Now, check if dropout is enabled
            if self.use_dropout:
                # TODO: Check dropout parameter when implementing dropout
                drop_param = self.dropout_param[layer]
                drop_out,drop_cache = dropout_forward(relu_out,drop_param)
                cache_dict['dropout'] = drop_cache


            #Update the caches for this layer
            cache_dict['affine'] = affine_cache
            cache_dict['relu'] = relu_cache
            # And then operate the global cache storage
            caches[layer+1] = cache_dict

            #And finally, route the output to the next iteration
            if self.use_dropout:
                # The last layer will be the dropout if it's activated
                current_input = drop_out
            else:
                # Else, the ReLU output should be passed forward
                current_input = relu_out



        # After we process all layers, the last layer should have the scores
        final_layer_weights = self.params["W"+str(self.num_layers)]
        final_layer_bias = self.params["b"+str(self.num_layers)]
        scores,first_cache = affine_forward(current_input,final_layer_weights,final_layer_bias)



        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        #Calculating the loss
        loss,softmax_dx= softmax_loss(scores,y)

        # Backpropagating the result of the very last affine layer of this model
        incoming_dx,incoming_dw,incoming_db = affine_backward(dout= softmax_dx,cache = first_cache)

        # Update the grads of the last layer
        grads["W"+str(self.num_layers)] = incoming_dw
        grads["b"+str(self.num_layers)] = incoming_db

        # Now, do the backward pass :P
        for layer in reversed(range(1,self.num_layers)):
            relu_input = None
            # If the net uses dropout
            if self.use_dropout:
                drop_out = dropout_backward(dout=incoming_dx, cache = caches[layer]["dropout"])
                relu_input = drop_out
                #TODO: FIX DROPOUT GRADIENTS HERE
            else:
                relu_input = incoming_dx

            # Backpropagating into the relu layer
            relu_out = relu_backward(dout=relu_input,cache = caches[layer]["relu"])

            # If the net uses batch normalization
            affine_input = None
            if self.use_batchnorm:
                batch_out = batchnorm_backward(dout=relu_out, cache = caches[layer]["batchnorm"])
                affine_input = batch_out
                #TODO: FIX BATCH NORM GRADIENTS HERE
            else:
                affine_input = relu_out

            #Passing through the last connected layer
            dx,dw,db = affine_backward(dout=affine_input,cache = caches[layer]['affine'])

            #Now, regularize the weights for the fully connected layer
            forward_weights = caches[layer]["affine"][1]
            dw += self.reg*forward_weights


            #And update the gradients for the weights and biases
            grads['W'+str(layer)] = dw
            grads['b'+str(layer)] = db

            # Regularize the loss
            loss += 0.5*self.reg*np.sum(self.params['W'+str(layer)]*self.params['W'+str(layer)])

            # And pipe over the input to the next iteration
            incoming_dx = dx

            # Done, yay


            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
