import numpy as np
import math
import time
import sys
import tensorflow as tf
import para
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxint)


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
p =RBM(794, 200, 0.1, weight_matrix, np.array(para.a), np.array(para.b))

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
piv = 0.1
for i in range(len(mnist.test.images)/slow_down):
	a= np.append(mnist.test.images[i], np.zeros(10))
	t= p.forward(a)
	plotData = p.back(t)
	predict = np.array([rounder(x, piv) for x in plotData[784:]])
	actual = mnist.test.labels[i]
	print i
	if np.argmax(predict) == actual:
		correct+=1
		
	total+=1

print float(correct)/float(total)
plt.plot(piv, float(correct)/float(total), '-o')
#e+=[error]
#errtot += error
#add a breakdown of which numbers are problemsome
print "Correct: ", correct, "/", total, "Pivot:", piv
print ("Run Time:", round(time.time()-beg, 2))
#plt.show()
