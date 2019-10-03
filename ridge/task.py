import numpy as np
from utils import *
import time

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	X_prime = np.ones((len(X[:,0]),1))
	for i in range(1,len(X[0,:])):
		if isinstance(X[0,i], str):
			labels = one_hot_encode(X[:,i], list(set(X[:,i])))
			X_prime = np.append(X_prime, labels, axis=1)
		else:
			append_data = X[:, i]
			append_data = append_data[np.newaxis, :].transpose()
			append_data = (append_data - np.mean(append_data)) / np.sqrt(np.var(append_data, dtype=np.float64))
			X_prime = np.append(X_prime, append_data, axis=1)
	
	newX = X_prime

	return newX.astype(float), Y.astype(float)

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	return 2 * (np.matmul(X, W) - Y) + 2 * _lambda * W

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	W = np.ones((X.shape[1],1))
	obj = ridge_objective(X, Y, W, _lambda)
	
	XX = X.transpose() @ X
	XY = X.transpose() @ Y

	for i in range(max_iter):
		delta = grad_ridge(W, XX, XY, _lambda)
		 # = grad_ridge(W, pX, pY, _lambda)
		# print(np.linalg.norm(delta, 2))
		new_W = W - lr * delta
		if np.linalg.norm(new_W-W, 2) < epsilon:
			break
		W = new_W

	return W


def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	list_X = np.split(X, k)
	list_Y = np.split(Y, k)
	avg_SSE = []
	iteration = 0
	for _lambda in lambdas:
		# print(iteration)
		iteration += 1
		error = 0
		for i, val_X in enumerate(list_X):
			val_Y = list_Y[i]
			train_X = np.array([x for j in np.arange(k) if j != i for x in list_X[j]])
			train_Y = np.array([y for j in np.arange(k) if j != i for y in list_Y[j]])
			weight = algo(train_X, train_Y, _lambda)
			error = error + sse(val_X, val_Y, weight)
		avg_SSE.append(error / k)

	return avg_SSE

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	def objective(W, X, Y, _lambda):
		return np.sum(np.square(Y - np.matmul(X, W))) + _lambda * np.sum(np.abs(W))

	W = np.ones((X.shape[1],1))
	obj = objective(W, X, Y, _lambda)

	norm = np.sum(np.square(X), axis=0)
	update = np.sum((Y - (np.matmul(X, W))) * X, axis=0)
	delta_constant = np.matmul(X.transpose(), X)

	for i in np.arange(max_iter):
		for k in np.arange(X.shape[1]):
			new_val = update[k] + norm[k] * W[k]

			if new_val - _lambda / 2 > 0:
				new_val = new_val - _lambda / 2
			elif new_val + _lambda / 2 < 0:
				new_val = new_val + _lambda / 2
			else:
				new_val = 0
			
			old_val = W[k] + 1 - 1
			W[k] = new_val / (norm[k] + 1e-6)

			delta = delta_constant[:,k].transpose() * (W[k] - old_val)
			update = update - delta

	return W

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = np.linspace(11,13,20)
	# Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	print("{} {}".format(min(scores), lambdas[scores.index(min(scores))]))
	print(scores)
	plot_kfold(lambdas, scores)
