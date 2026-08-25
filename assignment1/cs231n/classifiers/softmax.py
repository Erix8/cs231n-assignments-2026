from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # Accumulate the gradient for this training example:
        #   dL/dW[:, j] = (p_j - 1[j == y_i]) * x_i
        # The "p[y[i]] -= 1" turns the softmax probability of the correct class
        # into the (p - one_hot) difference that appears in the gradient.
        p[y[i]] -= 1
        dW += np.outer(X[i], p)

    # average over the batch plus the L2 regularization term
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    # scores for every class for every training example: (N, C)
    scores = X.dot(W)
    # subtract the max per row for numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)

    # softmax probabilities: (N, C)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=1, keepdims=True)

    # cross-entropy loss averaged over the batch + L2 regularization
    correct_log_probs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_log_probs) / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Gradient: dL/dW = (1/N) * X^T @ (probs - one_hot(y)) + 2 * reg * W
    probs[np.arange(num_train), y] -= 1
    dW = X.T.dot(probs) / num_train + 2 * reg * W


    return loss, dW
