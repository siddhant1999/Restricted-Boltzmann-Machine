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
correct=0
total=0

slow_down=1
incor = np.zeros(10)

def rounder(x, piv):
	if x>piv:
		return 1.0
	return 0

#plt.axis([0, 1, 0, 1])
more=0
for i in range(len(mnist.test.images)/slow_down):
	a= np.append(mnist.test.images[i], [0,0,0,0])
	t= p.forward(a)
	plotData = p.back(t)
	predict = [rounder(x, 0.5) for x in plotData[784:]]
	actual = convert_to_nodes(mnist.test.labels[i])
	
	
	
	if predict == actual:
		correct+=1
	else:
		valuepr = predict[0]*8 + predict[1]*4 + predict[2]*2 + predict[3]*1
		if valuepr > 9:
			more+=1
			print more
		else:
			incor[int(valuepr)] +=1.0
		#print valuepr
		'''
		valuepr = predict[0]*8 + predict[1]*4 + predict[2]*2 + predict[3]*1
		print "Predicted:", valuepr
		pData = plotData[:784]
		first_image = pData
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()

		sec_image = mnist.test.images[i]
		sec_image = np.array(sec_image, dtype='float')
		pixels = sec_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()
		'''
		
	
	total+=1
print float(correct)/float(total)
#e+=[error]
#errtot += error
#add a breakdown of which numbers are problemsome
print "Correct: ", correct, "/", total
print incor
print ("Run Time:", round(time.time()-beg, 2))
plt.show()
