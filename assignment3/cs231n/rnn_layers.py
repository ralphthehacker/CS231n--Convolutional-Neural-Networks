import numpy as np

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
      Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
      activation function.

      The input data has dimension D, the hidden state has dimension H, and we use
      a minibatch size of N.

      Inputs:
      - x: Input data for this timestep, of shape (N, D).
      - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
      - prev_h: Hidden state from previous timestep, of shape (N, H)
      - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
      - b: Biases of shape (H,)

      Returns a tuple of:
      - next_h: Next hidden state, of shape (N, H)
      - cache: Tuple of values needed for the backward pass.
    """

    next_h, cache = None, None

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################

    # Multiply the previous hidden state with the hidden weights
    h_times_weights = np.dot(prev_h, Wh)  # Will have shape NxH

    # And multiply the input vector with the input for X
    x_times_weights = np.dot(x, Wx)  # Will have shape  N*H

    # Finally, calculate the tanh to get the next hidden state
    next_h = np.tanh(h_times_weights + x_times_weights + b)

    # And put the necessary variables on the cache
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state of shape (N,H)
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """

    # Unpacking the cache values
    x, prev_h, Wx, Wh, b, next_h = cache

    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################

    # Steps



    # Tanh derivative
    dhtanh = (1 - np.square(next_h)) * dnext_h

    # Backpropping into the Bias
    db = np.sum(dhtanh, axis=0)

    # Backpropping into the hidden state and its weights
    dWh = np.dot(prev_h.T, dhtanh)
    dprev_h = np.dot(dhtanh, Wh.T)

    # Backpropping into the input
    dWx = np.dot(x.T, dhtanh)
    dx = np.dot(dhtanh, Wx.T)

    # Yay, done!

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass. List where L[i] contains the cache for a timestep i
  """
    N, T, D = x.shape
    _, H = h0.shape
    h, cache = np.zeros((N, T, H)), []
    num_inputs = x.shape[1]

    # This variable contains the H for this iteration
    input_h = h0[:]
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above.                                                                     #
    ##############################################################################


    # process the sequence
    for timestep in range(num_inputs):
        # Do forward pass
        next_h, cur_cache = rnn_step_forward(x[:, timestep, :], input_h, Wx, Wh, b)
        h[:, timestep, :] = next_h

        # Update the weights
        input_h = next_h

        # Store the cache
        cache.append(cur_cache)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
      Compute the backward pass for a vanilla RNN over an entire sequence of data.

      Inputs:
      - dh: Upstream gradients of all hidden states, of shape (N, T, H)

      Returns a tuple of:
      - dx: Gradient of inputs, of shape (N, T, D)
      - dh0: Gradient of initial hidden state, of shape (N, H)
      - dWx: Gradient of input-to-hidden weights, of shape (D, H)
      - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
      - db: Gradient of biases, of shape (H,)
    """

    a = (cache[0][0].shape)
    D = a[1]
    N, T, H = dh.shape
    timesteps = T


    dx = np.zeros(shape=(N, T, D))
    dh0 = np.zeros(shape=(N, H))
    dWx = np.zeros(shape=(D, H))
    dWh = np.zeros(shape=(H, H))
    db = np.zeros(shape=(H,))

    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    ##############################################################################
    # Iterating through the timesteps
    for timestep in reversed(xrange(timesteps)):
        # Getting the cache for this iteration
        cur_cache = cache[timestep]

        # Then, compute the gradients for all the variables
        cur_dx, cur_dprevh, cur_dWx, cur_dWh, cur_db = rnn_step_backward(dh[:, timestep, :], cur_cache)


        # And add them up
        dx[:,timestep,:] += cur_dx
        dh0 = cur_dprevh
        dWx += cur_dWx
        dWh += cur_dWh
        db += cur_db

        # Finally, pass the gradient to the previous timestep
        dh[:,timestep-1,:] += dh0
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
    out, cache = None, None
    out = np.zeros(shape=(x.shape[0],x.shape[1],W.shape[1]))

    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This should be very simple.                                          #
    ##############################################################################
    cache = (x,W)
    # Just index the output with X, since X already contains the indices
    out  = W[x]

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
    x,W = cache


    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    dW = np.zeros(shape=W.shape)
    # Just index dout with the valid indices of the examples to get the necessary gradients
    np.add.at(dW,x,dout) # And add those values into dW


  ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
  A numerically stable version of the logistic sigmoid function.
  """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################

    # Finding the activation vector for the input
    activation_vector = np.dot(x,Wx) + np.dot(prev_h,Wh) + b

    # Separating the activations into 4 vectors
    input,forget, output, g = np.split(activation_vector, 4,axis=1)

    # Calculting the functions for every single parameter
    input_n = sigmoid(input)
    forget_n = sigmoid(forget)
    output_n = sigmoid(output)
    g_n = np.tanh(g)

    # Finally, update the previous hidden and cell states
    next_c = forget_n*prev_c + input_n*g_n
    next_c_n = np.tanh(next_c)
    next_h = output_n * next_c_n

    cache = (x,prev_h,prev_c,Wx,Wh,b,input, forget, output, g, input_n,forget_n, output_n, g_n, next_c, next_c_n, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
    # Unpacking the cache
    x,prev_h,prev_c,Wx,Wh,b,input, forget, output, g, input_n,forget_n, output_n, g_n, next_c, next_c_n, next_h = cache

    # Initializing the derivatives
    dx = np.zeros(shape = (x.shape))
    dprev_h = np.zeros(shape = (prev_h.shape))
    dprev_c = np.zeros(shape = (prev_c.shape))
    dWx = np.zeros(shape = (Wx.shape))
    dWh = np.zeros(shape = (Wh.shape))
    db = np.zeros(shape = (b.shape))

    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################

    # Setting the tanh of the cell state
    tanh_cell_n = next_c_n

    # Calculating the derivative of the output
    doutput_n = tanh_cell_n * dnext_h

    # Then, calculate the derivative of the cell state
    dcell = (1 - tanh_cell_n**2) * (output_n * dnext_h)

    # Calculate the derivative of the Forget gate nonlinearity
    dforget_n = (prev_c * dcell) * dnext_c

    # And the derivative of the previous hidden state
    dprev_c = forget_n * dcell * dnext_c

    # Calculating the derivative of the input gate
    dinput_n = g_n * dcell * dnext_c

    # and of the G gate
    dg_n = input_n * dcell * dnext_c

    # Now, backpropagate through the activation functions and get the derivatives of the gates

    dg = (1.-g_n**2)*g_n # G gate
    dinput =  (input_n*(1-input_n)) * dinput_n # Input gate
    dforget = (forget_n*(1-forget_n)) * dforget_n # Forget gate
    doutput = (output_n*(1-output_n)) * doutput_n # Output gate
    d_activations = np.hstack((dinput,dforget,doutput,dforget))#.reshape((dinput.shape[0],4*dinput.shape[1]))

    # Backpropagate to the original matrix multiply and weights
    db = np.sum(d_activations,axis=0) # Bias
    dx = np.dot(d_activations, Wx.transpose())
    dWx = np.dot(x.transpose(), d_activations)
    dprev_h = np.dot(d_activations, Wh.transpose())
    dWh = np.dot(prev_h.transpose(), d_activations)



    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print 'dx_flat: ', dx_flat.shape

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

