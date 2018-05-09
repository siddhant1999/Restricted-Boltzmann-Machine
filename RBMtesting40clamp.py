import numpy as np
import math
import time
import tensorflow as tf
import para
#numpy.set_printoptions(threshold=sys.maxint)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt


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


def convert_to_nodes(x):
	a=[]
	for i in range(len(x)):
		for j in range(4):
			a.append(x[i])
	return a


print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

beg = time.time()
weight_matrix = np.array(para.w)
p =RBM(824, 250, 0.25, weight_matrix, np.array(para.a), np.array(para.b))

error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0

slow_down=1

def rounder(x, piv):
	if x>piv:
		return 1.0
	return 0

#plt.axis([0, 1, 0, 1])


correct=0
total=0
piv = 0.5
for i in range(len(mnist.test.images)/slow_down):
	a= np.append(mnist.test.images[i], np.zeros(40))
	t= p.forward(a)
	plotData = p.back(t)
	predict = np.array([rounder(x, piv) for x in plotData[784:]])
	actual = convert_to_nodes(mnist.test.labels[i])
#print i, predict, actual, 
#print ("Experimental", predict)
#print ("Theoretical", convert_to_nodes(mnist.validation.labels[i]))
#plotData = plotData[:784]

	print i
	if np.array_equal(predict, actual):

		correct+=1
		#print "correct: ", correct
	#else:
	#	print "wrong"
	total+=1
#diff= np.array(actual)-np.array(predict)
#error= np.dot(diff,diff)
#print error
#print ("Diff", diff)
#t1+=[time.time()-beg]
print float(correct)/float(total)
plt.plot(piv, float(correct)/float(total), '-o')
#e+=[error]
#errtot += error
#add a breakdown of which numbers are problemsome
print "Correct: ", correct, "/", total, "Pivot:", piv
print ("Run Time:", round(time.time()-beg, 2))
#plt.show()
