import numpy as np
import random as rand
import matplotlib.pyplot as plt

rand.seed(1)
w0 = rand.uniform(-0.25, 0.25)
w1 = rand.uniform(-1, 1)
w2 = rand.uniform(-1, 1)

print('w0 = ' + str(w0))
print('w1 = ' + str(w1))
print('w2 = ' + str(w2))

w0new = rand.uniform(-1, 1)
w1new = rand.uniform(-1, 1)
w2new = rand.uniform(-1, 1)

print('w0new = ' + str(w0new))
print('w1new = ' + str(w1new))
print('w2new = ' + str(w2new))

def getOutput(x1, x2, w0, w1, w2):
	return w0 + w1*x1 + w2*x2

def training(s, eta, n, s0, s1, w0, w1, w2):

	print('initial weights are: \n' + 'w0 = ' + str(w0) + '\nw1 = ' + str(w1) + '\nw2 = ' + str(w2))

	missc_list = []
	epoch = 0
	while(True):
		epoch += 1
		#print(epoch)
		miscalculations = 0
		for x, y in s.T:
			if getOutput(x, y, w0, w1, w2) < 0 and [x, y] not in s0:
				miscalculations += 1
				w0 += eta
				w1 += eta*x
				w2 += eta*y
			elif getOutput(x, y, w0, w1, w2) >= 0 and [x, y] not in s1:
				miscalculations += 1
				w0 -= eta
				w1 -= eta*x
				w2 -= eta*y
		if miscalculations == 0:
			missc_list.append(0)
			break
		missc_list.append(miscalculations)

	print('eta = ' + str(eta) + '\n' + 'n = ' + str(n) + '\n' + 'final weights are: \n' + 'w0 = ' + str(w0) + '\nw1 = ' + str(w1) + '\nw2 = ' + str(w2))
	print(missc_list)
	plt.plot(missc_list, marker='.', linestyle='-', label = 'η = '+str(eta))
	plt.xlabel('epoch')
	plt.ylabel('missclassifications')
	plt.title('for n = ' + str(n))
	plt.legend()

def assignment(n, w0, w1, w2):
	s = np.empty([2, n])

	for i in range(n):
		s[0][i] = rand.uniform(-1, 1)
		s[1][i] = rand.uniform(-1, 1)

	print(w0)

	s0x = []
	s0y = []
	s1x = []
	s1y = []

	for i in range(n):
		if getOutput(s[0][i], s[1][i], w0, w1, w2) < 0:
			s0x.append(s[0][i])
			s0y.append(s[1][i])
		else:
			s1x.append(s[0][i])
			s1y.append(s[1][i])

	s0 = []
	s1 = []

	for i in range(len(s0x)):
		s0.append([s0x[i], s0y[i]])
	for i in range(len(s1x)):
		s1.append([s1x[i], s1y[i]])

	plt.plot(s[0], (-s[0]*w1 - w0)/w2, linestyle ='solid', label = 'boundary')

	plt.plot(s0x, s0y, 'r.', label = 's0')

	plt.plot(s1x, s1y, 'g.', label = 's1')

	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('Plot for '+str(n)+' values')
	plt.legend()
	plt.show()

	print(s.T.shape)

	training(s, 1, n, s0, s1, w0new, w1new, w2new)
	training(s, 10, n, s0, s1, w0new, w1new, w2new)
	training(s, 0.1, n, s0, s1, w0new, w1new, w2new)



assignment(100, w0, w1, w2)
plt.show()
assignment(1000, w0, w1, w2)
plt.show()