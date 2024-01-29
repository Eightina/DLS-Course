import struct
import numpy as np
import gzip
import math
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename:str, label_filename:str) -> tuple:
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """        
    ### BEGIN YOUR CODE
    # get filename
    new_paths = []
    for path in [image_filename, label_filename]:
        new_paths.append(path[:path.find(".gz")])
    
    # decompressing
    for path in new_paths:        
        with gzip.GzipFile(filename=path+".gz", mode='rb') as uzf:
            with open(file=path, mode = "wb") as wf:
                wf.write(uzf.read())
            print('decompression done')
    
    # reading X      
    with open(file=new_paths[0], mode='rb') as uzx:
        mg_num = struct.unpack(">i", uzx.read(4))[0]
        num_examples = struct.unpack(">i", uzx.read(4))[0]
        height = struct.unpack(">i", uzx.read(4))[0]
        width = struct.unpack(">i", uzx.read(4))[0]
        input_dim = height * width
        print(mg_num, num_examples, height, width, input_dim)
        
        res_X = np.ndarray(shape=(num_examples, input_dim), dtype=np.dtype(np.float32))
        temp_fmt = ">" + "B" * input_dim
        for i in range(num_examples):
            res_X[i] = struct.unpack(temp_fmt, uzx.read(input_dim))
            
        # normalizing 
        res_X = res_X / 255.0
        # cur_min = res_X.min()
        # res_X = (res_X - cur_min) / (cur_max - cur_min)
        # res_X = res_X / ( - res_X.min())
        # res_X = res_X - res_X.min()
    
    # reading y
    with open(file=new_paths[1], mode='rb') as uzy:        
        mg_num = struct.unpack(">i", uzy.read(4))[0]
        num_labels = struct.unpack(">i", uzy.read(4))[0]
        print(mg_num, num_labels)
        
        temp_fmt = ">" + "B" * num_labels
        res_y = np.array(struct.unpack(temp_fmt, uzy.read(num_labels)), dtype=np.dtype(np.uint8))

        
    return (res_X, res_y)    

    ### END YOUR CODE


def softmax_loss(Z:np.ndarray, y:np.ndarray) -> float:
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # res = 0.0
    # length = Z.shape[0]
    # k = Z[0].shape[0]

    
    # a = np.log(np.sum(np.exp(Z), axis=1))
    # for i in range(k):
    #     Z[:, i] = a - Z[:, i]
        
    # Iy = np.zeros((length, k), dtype=np.dtype(np.float32))
    # for i in range(length):
    #     Iy[i][y[i]] = 1.0    
    # Z *= Iy
    
    # res = np.mean(np.sum(Z, axis=1))
    
    # return res

    # better:
    Z_y = Z[np.arange(Z.shape[0]), y]
    Z_sum = np.log(np.exp(Z).sum(axis=1))
    return np.mean(Z_sum - Z_y)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes) 
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    itr = math.ceil(X.shape[0] / batch)
    k = theta[0].shape[0]
    
    for i in range(itr):
        
        start = i * batch
        end = (i + 1) * batch
        if (end <= X.shape[0]):
            cur_batch = batch
        else:
            end = X.shape[0]
            cur_batch = end - start
        cur_X = X[start : end]
        cur_y = y[start : end]
        
        Z = np.exp(np.matmul(cur_X, theta))
        Z = Z / Z.sum(axis = 1).reshape(cur_batch, 1)
            
        Iy = np.zeros_like(Z)
        Iy[np.arange(Z.shape[0]), cur_y] = 1
        # print(cur_X.T)
        # print(Z - Iy)
        debug_temp = np.matmul(cur_X.T, Z - Iy)
        gradient = lr / cur_batch * debug_temp # id x ne * ne x nc = id x nc
        
        theta[:,:] -= gradient
        # printmat(theta)
        
    return
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim). ne x id
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,) ne
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim) id x hd
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes) hd x nc
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch 

    Returns:
        None
    """
    ### BEGIN YOUR CODE、
    def relu(x:np.ndarray):
        return np.maximum(x, 0)
    
    def norm(x: np.ndarray):
        return x / x.sum(axis = 1).reshape((x.shape[0],1))
        
    def binary(x:np.ndarray):
        return x.astype(np.bool8).astype(np.float32)
        
    itr = math.ceil(X.shape[0] / batch)
    
    for i in range(itr):
        
        start = i * batch
        end = (i + 1) * batch
        if (end <= X.shape[0]):
            cur_batch = batch
        else:
            end = X.shape[0]
            cur_batch = end - start
        cur_X = X[start : end] # ne x id
        cur_y = y[start : end] # ne
        
        Z1 = relu(cur_X @ W1) # ne x hd
        
        G2 = norm(np.exp(Z1 @ W2)) # ne x nc
        Iy = np.zeros_like(G2) 
        Iy[np.arange(cur_batch), cur_y] = 1
        G2 -= Iy
        
        G1 = np.multiply(binary(Z1), (G2 @ W2.T))
        
        gradient_W1 = 1 / cur_batch * (cur_X.T @ G1)
        gradient_W2 = 1 / cur_batch * (Z1.T @ G2)

        W1 -= lr * gradient_W1
        W2 -= lr * gradient_W2
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
