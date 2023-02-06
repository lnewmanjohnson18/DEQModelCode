import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, gradientHolder = None, **kwargs):
        super().__init__()
        # the internal function that dictates the dynamics of the DEQ
        self.f = f
        # the equilibrium point solver scheme
        self.solver = solver
        self.kwargs = kwargs
        # DEP
        self.gradientHolder = gradientHolder
        # DEP
        self.num_iter_capped = 0
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res, is_capped = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        
        # is_capped returns as None for acceleration schemes that don't have 
        # iteration cap tracking implemented yet
        if is_capped is not None:
            self.num_iter_capped += is_capped
        z = self.f(z,x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res, is_capped = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)

            return g
                
        z.register_hook(backward_hook)
        return z

    def clearIterCapped(self):
        self.num_iter_capped = 0
        return 0

    def getNumIterCapped(self):
        return self.num_iter_capped

def anderson2d(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    # Code (mostly) taken from implicit-layers-tutorial.org Ch . 4
    # W is dimension of the data

    """ Anderson acceleration for fixed point iteration. """
    bsz, W = x0.shape
    X = torch.zeros(bsz, m, W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    # have to return none for iter_capped here because I haven't implemented 
    # tracking iteration cap detection for this root solver
    return X[:,k%m].view_as(x0), res, None

def anderson1d(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0, plot_path = False):
    # Code (mostly) taken from implicit-layers-tutorial.org Ch . 4

    """ Anderson acceleration for fixed point iteration. """
    # print("x0")
    # print(x0)
    bsz = x0.shape[0]
    # print("bsz")
    # print(bsz)
    X = torch.zeros(bsz, m, 1, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, 1, dtype=x0.dtype, device=x0.device)
    # print("X")
    # print(X)

    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        # print("F[:,k%m]")
        # print(F[:,k%m])
        # input()
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    # return none for iter_capped here because I haven't implemented 
    # tracking iteration cap detection for this root solver
    return X[:,k%m].view_as(x0), res, None

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    # Code mostly taken from implicit-layers-tutorial.org Ch. 4

    is_capped = 0

    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (k+1) == max_iter:
            # print("HIT MAX ITERATIONS IN ANDERSON")
            # print ("\033[A                             \033[A")
            is_capped = 1
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res, is_capped

def andersonForLyapLoss(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    # Code mostly taken from implicit-layers-tutorial.org Ch. 4

    is_capped = 0

    """ Anderson acceleration for fixed point iteration. """
    num_vec, bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (k+1) == max_iter:
            # print("HIT MAX ITERATIONS IN ANDERSON")
            # print ("\033[A                             \033[A")
            is_capped = 1
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res, is_capped

class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

class MoonsResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.convDown = nn.Conv2d(n_channels, n_channels, kernel_size = 4, stride =4)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x): 

        print("x")
        print(x)
        print("x.shape")
        print(x.shape)
        input()
        x = self.upsample(x)
        z = self.upsample(z)
        
        y = self.norm1(F.relu(self.conv1(z)))

        return self.convDown(self.norm3(F.relu(z + self.norm2(x + self.conv2(y)))))

class LinearLayer(nn.Module):
    def __init__(self, n_features, n_inner_features):
        super().__init__()
        self.linear1 = nn.Linear(n_features, n_inner_features).double()
        self.linear2 = nn.Linear(n_inner_features, n_features).double()
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.linear1.weight.data.normal_(0.5, 0.1)
        self.linear2.weight.data.normal_(0.5, 0.1)

    def forward(self, z, x): 

        return x + F.relu(self.linear2(F.relu(self.linear1(z))))



def filterOutGTOne(dataset):
    indices = [i for i in range(len(dataset)) if (dataset[i][1] == 0) or (dataset[i][1] == 1)]
    return torch.utils.data.Subset(dataset, indices)

class BullsEye(Dataset):
    def __init__(self, num_samples = 500):
        self.num_samples = num_samples
        self.data = self.generateBullsEye(num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return(self.data[idx])


    def generateBullsEye(self, num_samples):
        plot_x_0 = []
        plot_x_1 = []
        plot_y_0 = []
        plot_y_1 = []
        data = []
        for i in range(num_samples):
            x = random.random()*2
            y = random.random()*2

            if ( ((x-1)*(x-1) + (y-1)*(y-1)) - .5 > 0 ):
                target = 1
                plot_x_1.append(x)
                plot_y_1.append(y)
            else:
                target = 0
                plot_x_0.append(x)
                plot_y_0.append(y)
            data.append(([x, y], target))

        plt.scatter(plot_x_0, plot_y_0, color='red')
        plt.scatter(plot_x_1, plot_y_1, color='blue')
        plt.grid()
        plt.show() 
        return data

class TwoMoons(Dataset):
    def __init__(self, num_samples = 500):
        self.num_samples = num_samples
        self.samples = self.generateMoons(num_samples)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
         return(self.samples[idx])

    def generateMoons(self, num_samples):
        # TODO there has to be a way to make this push out more convenient form
        # to avoid massive tensor fuckery in the future
        # Left Moon = Class 0 
        # Right Moon = Class 1
        data = []
        # plot_x_0 = []
        # plot_x_1 = []
        # plot_y_0 = []
        # plot_y_1 = []

        for i in range(num_samples):
            x = random.random()*3
            if (x < 1):
                # plot on left moon
                y = 2 - np.sqrt(2)*np.sqrt(2*x - x*x)
                target = 0
                # plot_x_0.append(x)
                # plot_y_0.append(y)
            elif (x > 2):
                # plot on right moon
                y = 1 + np.sqrt(2)*np.sqrt(-3 + 4*x - x*x)
                target = 1
                # plot_x_1.append(x)
                # plot_y_1.append(y)
            else:
                # plot on either moon
                if random.random() > .5:
                    y = 2 - np.sqrt(2)*np.sqrt(2*x - x*x)
                    target = 0
                    # plot_x_0.append(x)
                    # plot_y_0.append(y)
                else:
                    y = 1 + np.sqrt(2)*np.sqrt(-3 + 4*x - x*x)
                    target = 1
                    # plot_x_1.append(x)
                    # plot_y_1.append(y)
            data.append(([x,y], target))
        # plt.scatter(plot_x_0, plot_y_0, color='red')
        # plt.scatter(plot_x_1, plot_y_1, color='blue')
        # plt.grid()
        # plt.show()    

        # print(data)
        return data

        def generateMoonsOLD(self, num_samples):
            # TODO there has to be a way to make this push out more convenient form
            # to avoid massive tensor fuckery in the future
            # Left Moon = Class 0 
            # Right Moon = Class 1
            data = []
            # plot_x_0 = []
            # plot_x_1 = []
            # plot_y_0 = []
            # plot_y_1 = []

            for i in range(num_samples):
                x = random.random()*3
                if (x < 1):
                    # plot on left moon
                    y = 2 - np.sqrt(2)*np.sqrt(2*x - x*x)
                    target = 0
                    # plot_x_0.append(x)
                    # plot_y_0.append(y)
                elif (x > 2):
                    # plot on right moon
                    y = 1 + np.sqrt(2)*np.sqrt(-3 + 4*x - x*x)
                    target = 1
                    # plot_x_1.append(x)
                    # plot_y_1.append(y)
                else:
                    # plot on either moon
                    if random.random() > .5:
                        y = 2 - np.sqrt(2)*np.sqrt(2*x - x*x)
                        target = 0
                        # plot_x_0.append(x)
                        # plot_y_0.append(y)
                    else:
                        y = 1 + np.sqrt(2)*np.sqrt(-3 + 4*x - x*x)
                        target = 1
                        # plot_x_1.append(x)
                        # plot_y_1.append(y)
                # plot_x.append(x)
                # plot_y.append(y)
                data.append(([x,y], target))
                # plt.scatter(plot_x_0, plot_y_0, color='red')
            # plt.scatter(plot_x_1, plot_y_1, color='blue')
            # plt.grid()
            # plt.show()    

            print(data)
            return data

class OneDimensionalDataset(Dataset):
    def __init__(self, num_samples = 1000):
        self.num_samples = num_samples
        self.samples = self.generateData(num_samples)

    def generateData(self, num_samples):
        # NOTICE: the way the data is constructed right now is intended to make the torch.utils.data class easy to inherit
        # this means that the data is returned as a list of tuples and will have to be processed into a tensor somewhere
        # further down the pipeline 
        dataset = []

        for i in range(num_samples):
            # produce an x in [-1,1]
            x = (random.random()*2)-1
            # calculate f(x)
            # f(x) = 1.5x^3 + x^2 - 5x + 2sin(x) - 3 + noise
            y = (1.5*x**3) + x**2 - 5*x + 2*math.sin(x) - 3 + np.random.normal(0, 0.005)
            dataset.append((x, y))

        return dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print("==== PrintLayer ====")
        print("x")
        print(x)
        print("x.shape")
        print(x.shape)
        print("input to procede")
        input()
        return x

def SumLoss(target, dim=0):
    # returns loss == sum of tensor 'target' along dimension 'dim'
    return target.sum(dim=dim)

def clipWeights(net):
    # clips the weights of the given net to between [-1, 1]
    if hasattr(net, 'weight'):
        weights = module.weight.data
        weights = weights.clamp(-1,1)