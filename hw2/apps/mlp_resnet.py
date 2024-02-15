import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear0 = nn.Linear(in_features=dim, out_features=hidden_dim)
    norm0 = norm(dim=hidden_dim)
    relu0 = nn.ReLU()
    drop0 = nn.Dropout(p=drop_prob)
    linear1 = nn.Linear(in_features=hidden_dim, out_features=dim)
    norm1 = norm(dim=dim)
    relu1 = nn.ReLU()
    
    part1 = nn.Sequential(linear0, norm0, relu0, drop0, linear1, norm1)
    part2 = nn.Residual(part1)
    return nn.Sequential(part2, relu1)
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear0 = nn.Linear(in_features=dim, out_features=hidden_dim)
    relu0 = nn.ReLU()
    resblocks = [ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)]
    linear1 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
    return nn.Sequential(linear0, relu0, *resblocks, linear1)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    model: nn.Module
    
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    (error, loss, batchs, size) = (0, 0, len(dataloader.ordering), len(dataloader.dataset))
    loss_func = nn.SoftmaxLoss()
    if opt:
        model.train()
    else:
        model.eval()
    for (X, y) in dataloader:
        if opt:
            opt.reset_grad()
            
        cur_pred = model(X)
        cur_loss = loss_func(cur_pred, y)
        
        if opt:
            cur_loss.backward()
            opt.step()
        
        error += (cur_pred.numpy().argmax(axis=1) != y.numpy()).sum()
        loss += cur_loss.numpy() 
        
    return (error / size, loss / batchs)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_data = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size)
    test_loader = ndl.data.DataLoader(test_data, batch_size)

    model = MLPResNet(28 * 28, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        test_acc, test_loss = epoch(test_loader, model)

    return (train_acc, train_loss, test_acc, test_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
