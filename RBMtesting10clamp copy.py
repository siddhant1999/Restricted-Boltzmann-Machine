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

	def process(self, v):
		h= self.forward(v) #this is the first hidden output
		vPrime = self.back(h) #reconstructed input
		hPrime = self.forward(vPrime) #reconstructed output

		vTran = np.array([v]).transpose() #just a transpose
		posGrad = vTran*np.array([h]) 

		vPrimeTran = np.array([vPrime]).transpose()#just a transpose
		negGrad = vPrimeTran*np.array([hPrime])

		self.w= np.array(self.w[:len(v), :len(v)]) + self.learning_rate*(posGrad-negGrad)
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

beg = time.time()
weight_matrix = np.array(para.w)
p =RBM(794, 390, 0.4, weight_matrix, np.array(para.a), np.array(para.b))

error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0

for i in range(50000):
	if i%1000 == 0:
		print i
	t =mnist.train.images[i]
	l = mnist.train.labels[i]
	if l[5] != 1:
		continue
	yy = np.append(t,l)
	p.process(yy)

slow_down=1
t =mnist.train.images[30]
for i in range(50000):
	pp =mnist.train.images[i]
	l = mnist.train.labels[i]
	if l[5] == 1:
		t=pp
		break
first_image = np.array(t, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')

plt.show()

for i in range(30):
	kk= [0,0,0,0,0,0,0,0,0,0]
	a= np.append(t, kk)
	ttt= p.forward(a)
	plotData = p.back(ttt)
	first_image = np.array(plotData[:784], dtype='float')
	pixels = first_image.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.title(i)
	t=plotData[:784]
	plt.show()

exit()

def sigmoid(x):
		try:
			aaa = math.exp(-x)
		except:
			aaa = 0
		return 1.0 / (1.0 + aaa)
#plt.axis([0, 1, 0, 1])
for i in range(10):
	kk= [0,0,0,0,0,0,0,0,0,0]
	kk[i]=1

	a= np.append(np.zeros(784), kk)
	t= p.forward(a)
	plotData = p.back(t)
	#ttt= plotData*10000000
	#print ttt
	#print np.amax(plotData[:784])
	maxv = np.amax(plotData[:784])
	minv = np.amin(plotData[:784])
	for l in range(len(plotData[:784])):
		plotData[l] = (plotData[l]-minv)/(maxv-minv)
		plotData[l] = sigmoid(plotData[l])

	#print np.amax(plotData[:784]) 
	print i
	first_image = np.array(plotData[:784], dtype='float')
	pixels = first_image.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.title(i)
	#print plotData
	plt.show()
