import numpy as np
import math
import time
import tensorflow as tf
import parameters
#numpy.set_printoptions(threshold=sys.maxint)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt


def weights_file():
	file=open("testing123.txt","r")
	a=file.read().split('[')
	b=a[-2].split(' ')#maybe should be a space
	#print b
	
	final =[]
	for j in range(1, len(a)):
		b=a[j].split(',')
		temp=[]
		for i in range(len(b)):
			b[i] =b[i].strip()
			if len(b[i]) < 1:
				continue
			if b[i][-1] == ']':
				b[i]=b[i][0:-1]
			
			#print "::", b[i], "::"
			temp.append(float(b[i]))
		final.append(temp)
			#print float(b[i])
	
	#print final
	#print np.shape(np.array(final))
				#print float(b[i])
	return final


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
	num, = np.where(x==1)
	num=num[0]
	a = num/8
	num-=8*a
	b=(num)/4
	num-=4*b
	c=(num)/2
	num-=2*c
	d=num
	return [a,b,c,d]


print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

beg = time.time()
weight_matrix = weights_file()
p =RBM(788, 160, 0.27, weight_matrix, parameters.a, parameters.b)

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


for iv in range(1,10):
	correct=0
	total=0
	piv = iv/10.0
	for i in range(len(mnist.test.images)/slow_down):
		a= np.append(mnist.test.images[i], [0,0,0,0])
		t= p.forward(a)
		plotData = p.back(t)
		predict = [rounder(x, piv) for x in plotData[784:]]
		actual = convert_to_nodes(mnist.test.labels[i])
	#print i, predict, actual, 
	#print ("Experimental", predict)
	#print ("Theoretical", convert_to_nodes(mnist.validation.labels[i]))
	#plotData = plotData[:784]
	
		if predict == actual:
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
plt.show()
