import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
from numba import vectorize

@vectorize(['float32(float32)'], target='cuda')
def tre(a):
	return a+1

def abcd(x):
	return 1.0 / (1.0 + math.exp(-x))

@staticmethod
def sig(x):
	return abcd(x)
	#return 1.0 / (1.0 + math.exp(-x))

class RBM():
	#learning_rate=1
	aq =0

	sigmoid = sig
	def __init__(self, rows, cols, rate):
		self.w= np.array(np.random.uniform(low=-1, high=1, size=(rows, cols)))
		self.a= np.array(np.random.uniform(low=-1, high=1, size=(rows)))#input bias
		self.b= np.array(np.random.uniform(low=-1, high=1, size=(cols)))#output bias
		self.learning_rate = rate
	'''
	def sigmoid(self, x):
		try:
			aaa = math.exp(-x)
		except:
			aaa = 0
		return 1.0 / (1.0 + aaa)
	'''
	def process(self, v):
		h= self.forward(v) #this is the first hidden output
		vPrime = self.back(h) #reconstructed input
		hPrime = self.forward(vPrime) #reconstructed output

		vTran = np.array([v]).transpose() #just a transpose
		posGrad = vTran*np.array([h]) 

		vPrimeTran = np.array([vPrime]).transpose()#just a transpose
		negGrad = vPrimeTran*np.array([hPrime])

		#print "array", np.shape(np.array(self.w[:len(v), :len(v)]))

		self.w= np.array(self.w[:len(v), :len(v)]) + self.learning_rate*(posGrad-negGrad)#updating the weight
		self.a=np.array(self.a[:len(v)])+ self.learning_rate*(np.array(v)-np.array(vPrime))
		self.b=np.array(self.b[:len(v)]) + self.learning_rate*(np.array(h)-np.array(hPrime))
		

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
print "Here: ", tre(123)
#p= RBM(784, 5)
beg = time.time()
error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0
for ttt in range(100, 780, 200):
	for rrr in range(1, 5):
		rrrate=float(rrr)/5.0 
		p= RBM(788, ttt, rrrate)
		count = 0
		val = 5
		slow_down=100;
	
		print ttt, rrrate,
		#Training

		for i in range(len(mnist.train.images)/slow_down):
			s = convert_to_nodes(mnist.train.labels[i])
	
			first_image = mnist.train.images[i]
			#if i%100==0:
				#print i
			#print count, i
			a=np.append(mnist.train.images[i], s)
			p.process(a)
			count+=1
		#Testing

		#t1=[]
		#e=[]
		errtot = 0
		for i in range(len(mnist.validation.images)/slow_down):
			a= np.append(mnist.validation.images[i], [0,0,0,0])
			t= p.forward(a)
			plotData = p.back(t)
			predict = [round(x,2) for x in plotData[784:]]
			actual = convert_to_nodes(mnist.validation.labels[i])
			#print ("Experimental", predict)
			#print ("Theoretical", convert_to_nodes(mnist.validation.labels[i]))
			#plotData = plotData[:784]
	

			diff= np.array(actual)-np.array(predict)
			error= np.dot(diff,diff)
			#print error
			#print ("Diff", diff)
			#t1+=[time.time()-beg]
			#plt.plot(time.time()-beg, error)
			#e+=[error]
			errtot += error
		print errtot
		if errtot<besterror:
			besterror = errtot
			bestrate = p.learning_rate
			bestnodes = ttt
	
		error_rates+=[errtot]
		#plt.plot(t1, e, 'k')
print "Best:"
print besterror, bestnodes, bestrate
print error_rates
print ("Run Time:", round(time.time()-beg, 2))
plt.show()
