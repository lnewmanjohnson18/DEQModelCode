import sys
import os
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import gradcheck
import importlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

###########################
# Option Select
###########################

NUM_MODELS = 8

# If no options were passed remind to select
if len(sys.argv) == 1:
	print("Must select 'server' or 'non-server'")
	quit() 

# If 'server' is selected; run with threads and disable plot display
if sys.argv[1] == "server":
	print("Server Options Selected")
	NUM_WORKERS = 8
	# has to be done here so .use() can be called between imports
	# of matplotlib and matplotlib.pyplot
	import matplotlib
	# allows plot creation without display
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

# If non-server no workers and matplotlib is used normally
if sys.argv[1] == "non-server":
	print("Non-Server Options Selected")
	NUM_WORKERS = 0
	# has to be done here so .use() can be called between imports
	# of matplotlib and matplotlib.pyplot in the server options select
	import matplotlib
	import matplotlib.pyplot as plt

# save reference to correct device to indicate location for later operations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
deq_utils = importlib.import_module('deqUtils')

###########################
# Dataloaders
###########################

BATCH_SIZE = 100

cifar10_train = datasets.CIFAR10(
	root = './data',
	train = True,
	download = True,
	transform = transforms.ToTensor())

train_loader = DataLoader(cifar10_train,
	batch_size = BATCH_SIZE,
	shuffle = True,
	num_workers = NUM_WORKERS)

cifar10_test = datasets.CIFAR10(
	root = './data',
	train = False,
	download = True,
	transform = transforms.ToTensor())

test_loader = DataLoader(cifar10_test,
	batch_size = BATCH_SIZE,
	shuffle = True,
	num_workers = NUM_WORKERS)


for iteration_num in range (NUM_MODELS):
	print("===TRAINING MODEL %d==="%iteration_num)

	###########################
	# Model Definition
	###########################

	# NOTE: many of the values in this section of the code are tuned specifically for CIFAR10. The chosen 
	# 		number of channels and inner_channels, num_groups, and the Conv2d values play an important
	#		role in the DEQ Layer's stability. 

	channels = 48
	inner_channels = 64

	# define the interior function that will be run repeatedly within the DEQ layer
	f = deq_utils.ResNetLayer(channels, inner_channels, kernel_size = 3, num_groups = 8)

	# define the DEQ layer
	DEQLayer = deq_utils.DEQFixedPoint(f, deq_utils.anderson, tol = 1e-3, max_iter = 25, m = 5)
	# NOTE: if using a dataset with a different number of dimensions you need the correct acceleration scheme
	#		either deq_utils.anderson2d or deq_utils.anderson1d
	# NOTE: when this is called as part of the model() function call later, it is important to realize
	# 		that the DEQFixedPoint layer has a custom forward function 

	# lay out the structure of the net
	model = nn.Sequential(

		# convolve the input and normalize
		nn.Conv2d(3, channels, kernel_size = 3, bias = True, padding = 1),
		nn.BatchNorm2d(channels),

		# run the DEQ Layer 
		DEQLayer,

		# normalize the DEQ output in the new space
		nn.BatchNorm2d(channels),
		nn.AvgPool2d(8, 8),
		
		# flatten and run through a final fully connected layer to classify 
		nn.Flatten(),
		nn.Linear(channels*4*4, 10)

		# NOTE: several elements are tailored to the CIFAR10 dataset
 		#		datasets with different dimensions will require a 
 		# 		different number of channels and inner channels.
 		#		If not in 3 dimensions a different anderson function
 		# 		needs to be called (found in deqUtils.py).


		).to(device)

	###########################
	# Training Loop
	###########################

	# function that will manage the learning process
	def training_loop(loader, model, opt = None, lr_scheduler = None):
		total_loss, total_err = 0., 0.

		# declare evaluation or training mode
		if opt is None:
			model.eval()
		else:
			model.train()
		counter = 0
		for X, y in loader: 
			# print("processing sample %d" %counter)
			X, y = X.to(device), y.to(device)
			pred = model(X)
			loss = nn.CrossEntropyLoss()(pred, y)
			if opt:
				# this shorthand simplifies a somewhat involved process:

				# first the optimizer is cleaned out with .zero_grad()
				opt.zero_grad()

				# then the backward function needs to be explicitly called on the calculated loss
				# this calls the custom backward_hook function defined in the DEQFixedPoint class
				# found in deqUtils.py
				loss.backward()

				# now that we have populated the relevant gradients .step() can now be called 
				# on both the optimizer and the scheduler
				opt.step()
				lr_scheduler.step()

			counter += 1
			total_err += (pred.max(dim=1)[1] != y).sum().item()
			total_loss += loss.item() * X.shape[0]

		return total_err / len(loader.dataset), total_loss / len(loader.dataset)

	###########################
	# Train 
	###########################

	opt = optim.Adam(model.parameters(), lr=1e-3)
	max_epochs = 35
	scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)

	for i in range(max_epochs):
		print("EPOCH %d" %i)
		print(training_loop(train_loader, model, opt, scheduler))
		print(training_loop(test_loader, model)) 

	###########################
	# Save 
	###########################

	torch.save(model.state_dict(), './deq_cifar10_models/model' + str(iteration_num))