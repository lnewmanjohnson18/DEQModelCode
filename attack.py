# NOTE: Edit the below import to replace the resNet that is going to be attacked.
#		Currently it will look for 'cifar10_models/resnet50.py' but any pytorch
#		model can be called in its place

from cifar10_models.resnet import resnet50
# import the resnet50 model from local directory
# taken from https://github.com/kuangliu/pytorch-cifar
 
# by default DEQ Model 1 is tested, if this is set to a number 1-9
# it will test that number model instead.
specific_model_number = None


import sys
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import importlib
import pickle
import numpy as np
import multiprocessing
from multiprocessing import Pool

# initialization of the script PRIO 1
def initialize():

	### option select

	# If no/invalid options were passed remind to select

	if len(sys.argv) != 4:
		print("all 3 arguments [server/non, model, output path] were not passed.")
		quit()

	if not (sys.argv[1]  == "server" or sys.argv[1] == "non-server"):
		print("First argument must select 'server' or 'non-server'")
		quit() 

	if not (sys.argv[2]  == "ResNet" or sys.argv[2] == "DEQ"):
		print("Second argument must select 'ResNet' or 'DEQ'")
		quit()

	# If server: have threads and won't try to display plot
	if sys.argv[1] == "server":
		print("Server Options Selected")
		NUM_WORKERS = 12 
		# has to be done here so .use() can be called between imports
		import matplotlib
		# allows plot creation without display
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

	# If non-server: no workers and matplotlib is imported normally
	if sys.argv[1] == "non-server":
		print("Non-Server Options Selected")
		NUM_WORKERS = 1
		# has to be done here so .use() can be called between imports below
		import matplotlib
		import matplotlib.pyplot as plt


	# Select which kind of model to attack

	# Initialize the ResNet if selected
	if sys.argv[2] == "ResNet":

		# the pretrained model from cifar10_models.renset imported above
		model = resnet50(pretrained=True)

		# no training should occur in this script
		model.eval()


	# Initialize the DEQ Model if selected
	if sys.argv[2] == "DEQ":

		# declare the parameters of the DEQ
		CHANNELS = 48
		INNER_CHANNELS = 64
		BATCH_SIZE = 1

		# import helper code from the deq utilities file
		deq_utils = importlib.import_module('deqUtils')

		# tell torch to use the right device
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		# load the function defining the iterative layer of the DEQ
		f = deq_utils.ResNetLayer(CHANNELS, INNER_CHANNELS, kernel_size = 3, num_groups = 8)

		# construct the full DEQ unit  
		DEQLayer = deq_utils.DEQFixedPoint(f, deq_utils.anderson, tol = 1e-3, max_iter = 25, m = 5)

		# lay out the structure of the net
		model = nn.Sequential(
		nn.Conv2d(3, CHANNELS, kernel_size = 3, bias = True, padding = 1),
		nn.BatchNorm2d(CHANNELS),
		DEQLayer,
		nn.BatchNorm2d(CHANNELS),
		nn.AvgPool2d(8, 8),
		nn.Flatten(),
		nn.Linear(CHANNELS*4*4, 10)
		).to(device)

		# load in DEQ model
		model_name = 'deq_cifar10_models/model1'

		# load specific model number if specified at top of file
		if specific_model_number is not None:
			model_name = 'deq_cifar10_models/model' + str(specific_model_number)

		model.load_state_dict(torch.load(model_name, map_location=device))

		# no training should occur in this script
		model.eval()

	output_path = sys.argv[3]

	# tell torch to use the right device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# make sure the model is stored on the right device
	model.to(device)

	BATCH_SIZE = 1

	cifar10_test = datasets.CIFAR10(
		root = './data',
		train = False,
		download = True,
		transform = transforms.ToTensor())

	test_loader = DataLoader(cifar10_test,
		batch_size = BATCH_SIZE,
		shuffle = False,
		num_workers = 0)


	return model, test_loader, device, output_path, NUM_WORKERS

def createJacobian(pred, device, X_hat):
	# takes the output of the model (the 10 elem array of probabilities)
	# and returns the jacobian representing change in that array w.r.t the 3072 inputs 
	jacobian = torch.zeros((3072, 10)).float()
	for i in range(10):
		grad_mask = torch.zeros((1, 10))
		grad_mask[0, i] = 1
		grad_mask = grad_mask.to(device)
		pred.backward(gradient = grad_mask, retain_graph = True)
		# this will create a gradient that only relates to the ith element
		# of the inputs, and is the ith column of the complete jacobian
		jacobian[:,i] = X_hat.grad.view(3072).float()
		X_hat.grad.zero_()
	return jacobian

################################################
# Catalogue and Prepare Dataset
################################################

# organize data in chunks convenient for the pool PRIO 2
def package_data():
	ID = 0
	packed_data = []
	for X, y in test_loader:
		X, y = X.to(device), y.to(device)
		# package dataas a tuple so it has an ID that will follow it around
		packed_data.append((ID, X, y))
		ID += 1
	return packed_data


################################################
# Define The Multi-Processing Workflow
################################################

def run_sample(data_tuple, device, model, parameter_tuple):

	ID, X, y = data_tuple

	STEP_SIZE, MAX_DELTA_SIZE, convergence_cutoff, max_steps = parameter_tuple

	# initialize a random perturbation as a starting point
	delta = (torch.rand(3072)*10e-4).view(3, 32, 32)
	# make sure it's stored on the right device
	delta = delta.to(device)

	# dummy starting values
	change_over_iteration = 1000
	prev = None
	
	step_counter = 0

	theta_x = model(X)

	previous_iterations = []

	while (change_over_iteration > convergence_cutoff and step_counter < max_steps):
		# starting the gradient descent process with parameters
		# change_over_iteration: determines when we've converged
		# step counter: maximum number of iterations allowed

		# clone so repeated operations do not build up in the diff graph
		X_clone = X.clone()

		X_hat = None
		X_hat = X_clone + delta
		# set to none to clean gradients on X_hat tensor. possibly unnecessary
		X_hat.requires_grad_(requires_grad=True)
		X_hat.retain_grad()

		theta_x_plus_delta = None
		theta_x_plus_delta = model(X_hat)
		# get the output of model post perturbation

		dy_hat_d_delta = createJacobian(theta_x_plus_delta, device, X_hat)
		# compute the Jacobian expression change in output post perturbation
		# w.r.t. change in perturbation

		# calculate outer chain rule term
		term_A = theta_x_plus_delta.clone().detach() - theta_x.clone().detach()
		term_A = term_A.to(device)

		# calculate inside term
		term_B = dy_hat_d_delta.clone().detach()
		term_B = term_B.to(device)

		# combine terms
		gradient = torch.matmul(term_B, torch.transpose(term_A, 0, 1))

		# take a step
		delta += -1*STEP_SIZE*gradient.view(3, 32, 32)

		# if attack is too large, i.e. beyond the budget: project back onto n-ball
		norm_of_delta = torch.linalg.norm(torch.linalg.matrix_norm(delta))
		# project back onto appropriate n-ball
		if (norm_of_delta) > MAX_DELTA_SIZE:
			delta = delta * (MAX_DELTA_SIZE / norm_of_delta)

		# save delta to compare later.
		previous_iterations.append(copy.deepcopy(delta))
		
		# check to see if the classifier is fooled by the current delta and 
		# stop the algorithm if it is
		if torch.argmax(theta_x_plus_delta) != torch.argmax(theta_x):
			break
		# NOTE: This actually tests against the delta after iteration n-1
		# not a big deal (just a constant +1 to all num_iteration values when we early stop)
		# and lets this sit and the bottom of the loop while not having to computer model(x_hat) twice


		# if 5 iterations have been run (which is the hard minimum), calculate change_over_iteration
		# which compares how delta has changed over the last 5 iterations and uses this
		# as a heuristic for having converged
		if len(previous_iterations) >= 5:
			change_over_iteration = 0
			for i in range (1,5):
				change_over_iteration += torch.linalg.norm(torch.linalg.matrix_norm(previous_iterations[-i] - previous_iterations[-(i+1)]))
			# INSIDE TERM: frobenius norm of the difference between the delta over iterations in the 32 x 32
			# OUTSIDE TERM: L2 norm of the resultant 1x3 vector from the inside term


		step_counter += 1

	print("sample number: %d completed with number of steps:"%ID)
	print(step_counter)

	result_data_tuple = (
						step_counter,
						copy.deepcopy(delta),
						copy.deepcopy(X),
						copy.deepcopy(y) 
						)
	# data tuple to be saved for the per sample results
	# of form: num_iterations, delta, X, y, theta_x, X_hat, y_hat
	# have to send to the CPU because CUDA communication between processes is illegal

	worker_res = {}
	worker_res["was_fooled"] = False
	if (torch.argmax(theta_x, 1) != torch.argmax(theta_x_plus_delta, 1)):
		print("WAS FOOLED")
		worker_res["was_fooled"] = True

	worker_res["ID"] = ID
	worker_res["data_tuple"] = result_data_tuple


	return worker_res


################################################
# Run Multi-Processing 
################################################

# organize job list PRIO 3
def organize_jobs():
	jobs = []
	parameter_tuple = STEP_SIZE, MAX_DELTA_SIZE, convergence_cutoff, max_steps 
	for i in range(len(packed_data)):
		job_tuple = (packed_data[i], device, model, parameter_tuple)
		jobs.append(job_tuple)
	return jobs


# worker funciton for the pool; makes each specific call
# code inside this function will be executed by the child process
def mp_worker(job_info):
	# recieves input as a tuple since map chunks up jobs and sends that way
	data_tuple, device, model, parameter_tuple = job_info
	# think of last 3 args as environment definitions
	return(run_sample(data_tuple, device, model, parameter_tuple))


# chunks up jobs and activates the workers
def mp_handler(NUM_WORKERS):
	pool = multiprocessing.Pool(NUM_WORKERS)
	res = pool.map(mp_worker, jobs)
	return res


if __name__ == "__main__":


	torch.multiprocessing.set_start_method('spawn')
	torch.multiprocessing.set_sharing_strategy('file_system')
	# this is necessary for using MP with cuda tensors
	# has to be done here and not with the other initialization procedures
	# because if called in initialize() the context is already declared.


	# set parameters for the attacker's gradient descent
	# set the lambda term
	STEP_SIZE = 10e-3

	# set the budget (constraint on the perturbation via its L2 size)
	# perturbation will fit inside an n-ball with radius MAX_DELTA_SIZE
	MAX_DELTA_SIZE = 1

	# the point at which we decide convergence has been reached
	convergence_cutoff = 1

	# the maximum number of steps the GD will do before abandoning
	max_steps = 1000

	# call functions to set up environment
	model, test_loader, device, output_path, NUM_WORKERS = initialize() # PRIO 1
	packed_data = package_data() # PRIO 2
	jobs = organize_jobs() # PRIO 3

	# calls handler function which chunks jobs and activates
	worker_res = mp_handler(NUM_WORKERS)

	parameter_dict = {
		'MAX_DELTA_SIZE':MAX_DELTA_SIZE,
		'max_steps':max_steps,
		'STEP_SIZE':STEP_SIZE,
		'convergence_cutoff':convergence_cutoff
		}

	# add the dictionary to the front of the list for safety's sake
	worker_res.insert(0, parameter_dict)

	# pickle and save the results dictionary into 'output_path'
	with open(output_path, 'wb') as f:
		pickle.dump(worker_res, f)
