import numpy as np
import random as rand
import idx2numpy
import matplotlib.pyplot as plt

rand.seed(1)

def step(input):
	return_array = np.empty(np.shape(input))
	for i in range(len(input)):
		if input[i] < 0:
			return_array[i] = 0
		else:
			return_array[i] = 1

	return return_array

def testing(w, testing_set, label_set):
	errors = 0

	for i in range(len(testing_set)):
		v = np.matmul(w, testing_set[i].flatten())
		if np.argmax(v) != label_set[i]:
			errors+=1

	print('percent of testing errors: ' + str((errors/len(testing_set)) * 100))

def training(eta, epsilon, n, training_set, label_set):
	w = np.random.rand(10, 784)
	#print(w)
	epoch = 0
	errors = []
	while epoch == 0 or (errors[epoch - 1]/n > epsilon and epoch < 75):
		
		errors.append(0)
		
		for i in range(n):
			#print(label_set[i])
			v = np.matmul(w, training_set[i].flatten())
			if np.argmax(v) != label_set[i]:
				errors[epoch] += 1
		
		print('epoch: '+str(epoch))
		print('curr epsilon: '+str(errors[epoch]/n))
		print(errors[epoch])
		epoch += 1

		for i in range(n):
			d = np.zeros([10, 1])
			d[label_set[i]] = 1
			#w = w + eta*(d + step(np.matmul(w, training_set[i].flatten())))
			#np.matmul(np.subtract(d, step(np.matmul(w, training_set[i].flatten()))), training_set[i].flatten().T)
			#print(np.shape(training_set[i].reshape([1, 784])))
			w = np.add(w, eta * np.matmul(np.subtract(d, step(np.matmul(w, training_set[i].flatten())).reshape([10, 1])), training_set[i].reshape([1, 784])))

	print(errors)
	plt.plot(errors)
	plt.xlabel('Epoch number')
	plt.ylabel('misclassifications')
	plt.title('for Eta = ' + str(eta) + ' and n = ' + str(n))
	plt.show()
	return w

training_set = idx2numpy.convert_from_file('C:/Users/visha/OneDrive/Documents/nn codes/mnist set/train-images.idx3-ubyte') #change this to your directory
label_set = idx2numpy.convert_from_file('C:/Users/visha/OneDrive/Documents/nn codes/mnist set/train-labels.idx1-ubyte') #change this to your directory
testing_set = idx2numpy.convert_from_file('C:/Users/visha/OneDrive/Documents/nn codes/mnist set/t10k-images.idx3-ubyte') #change this to your directory
testing_label_set = idx2numpy.convert_from_file('C:/Users/visha/OneDrive/Documents/nn codes/mnist set/t10k-labels.idx1-ubyte') #change this to your directory

#print(np.shape(training_set))

w = training(1, 0, 50, training_set, label_set)
testing(w, testing_set, testing_label_set)

w = training(1, 0, 1000, training_set, label_set)
testing(w, testing_set, testing_label_set)


#w = training(1, 0, len(training_set), training_set, label_set)
#testing(w, testing_set, testing_label_set)

w = training(0.1, 0.15, len(training_set), training_set, label_set)
testing(w, testing_set, testing_label_set)