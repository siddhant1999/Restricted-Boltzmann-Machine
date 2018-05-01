import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt


class RBM():
	learning_rate=1
	aq =0

	def __init__(self, rows, cols):
		self.w= np.array(np.random.uniform(low=-1, high=1, size=(rows, cols)))
		self.a= np.array(np.random.uniform(low=-1, high=1, size=(rows)))#input bias
		self.b= np.array(np.random.uniform(low=-1, high=1, size=(cols)))#output bias

	def sigmoid(self, x):
		try:
			aaa = math.exp(-x)
		except:
			aaa = 0
		return 1.0 / (1.0 + aaa)

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




print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

#p= RBM(784, 5)

#we will now encode the number being sent as the 4 last nodes
#perhaps to give them more importance in the future we can enode the values as bits to a larger set of nodes and average the result
p= RBM(788, 20)
count = 0
val = 5
slow_down=30;
beg = time.time()

#Training

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

for i in range(len(mnist.train.images)/slow_down):
	'''
	if not mnist.train.labels[i][val]:
		continue
	'''
		#because we are currently just training for a single character
	#print mnist.train.labels[i]
	s = convert_to_nodes(mnist.train.labels[i])
	
	
	first_image = mnist.train.images[i]
	print count, i
	a=np.append(mnist.train.images[i], s)
	p.process(a)
	#p.process(mnist.train.images[i])
	count+=1
#Testing
for i in range(len(mnist.train.images)):
	if mnist.train.labels[i][val]:
		#t = p.forward(mnist.train.images[i])
		t= p.forward(np.ones(788))
		#not entirely sure if I should be pushing through a 1s array of the image itself.
		plotData = p.back(t)
		#print round(plotData[784:], 2)
		print ("Experimental", [round(x,2) for x in plotData[784:]])
		print ("Theoretical", convert_to_nodes(mnist.train.labels[i]))
		plotData = plotData[:784]

		print ("Run Time:", round(time.time()-beg, 2))
		first_image = plotData
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()
		
		'''
		#first_image = mnist.train.images[i]
		first_image = np.zeros(784)
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()
		'''
		break
