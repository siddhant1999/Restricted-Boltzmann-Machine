import numpy as np
import math
import time
import tensorflow as tf
import para
#numpy.set_printoptions(threshold=sys.maxint)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt



#yeah this is actually a shit method



class RBM():
	#learning_rate=1
	aq =0

	def __init__(self, rows, cols, rate, w, a, b):
		self.w= np.array(w)
		self.a= np.array(a)#input bias
		self.b= np.array(b)#output bias
		self.learning_rate = rate

	def sigmoid(self, x):
		try:
			aaa = math.exp(-x)
		except:
			aaa = 0
		return 1.0 / (1.0 + aaa)

	def forward(self, x):
		summ = (np.matrix(self.w)).transpose() *np.matrix(x).transpose() + np.matrix(self.b).transpose()
		h = np.apply_along_axis(self.sigmoid, 1, summ)
		return h
		

	def back(self, h):
		summ =np.matrix(self.w)*np.matrix(h).transpose() + np.matrix(self.a).transpose()
		x = np.apply_along_axis(self.sigmoid, 1, summ)
		return x


print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

beg = time.time()
weight_matrix = np.array(para.w)
p =RBM(794, 390, 0.15, weight_matrix, np.array(para.a), np.array(para.b))

error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0

slow_down=1



#plt.axis([0, 1, 0, 1])


correct=0
total=0
for i in range(len(mnist.test.images)/slow_down):
	if i%100==0:
		print i
	ztz= [0,0,0,0,0,0,0,0,0,0]
	actual = mnist.test.labels[i]
	for j in range(10):
		z=[0,0,0,0,0,0,0,0,0,0]
		z[j] =1
		print z,
		a= np.append(mnist.test.images[i], z)
		t= p.forward(a)
		plotData = p.back(t)
		predict = np.array(plotData[784:])
		ztz[np.argmax(predict)]+=1
		print [ '%.2f' % elem for elem in predict ]
	z=[0,0,0,0,0,0,0,0,0,0]	
	a= np.append(mnist.test.images[i], z)
	t= p.forward(a)
	plotData = p.back(t)
	predict = np.array(plotData[784:])
	print z, [ '%.2f' % elem for elem in predict ]

	z=[1,1,1,1,1,1,1,1,1,1]	
	a= np.append(mnist.test.images[i], z)
	t= p.forward(a)
	plotData = p.back(t)
	predict = np.array(plotData[784:])
	print z, [ '%.2f' % elem for elem in predict ]

	if np.argmax(ztz) == np.argmax(actual):
		correct+=1
	print ztz, actual
	ppp=raw_input()
	total+=1
print float(correct)/float(total)


print "Correct: ", correct, "/", total
print ("Run Time:", round(time.time()-beg, 2))
#plt.show()
