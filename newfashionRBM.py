import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt

beg = time.clock()
class RBM():
	learning_rate=0.5
	aq =0

	def __init__(self, length):
		self.w= np.array(np.random.uniform(low=-1, high=1, size=(length, length)))
		self.a= np.array(np.random.uniform(low=-1, high=1, size=(length)))#input bias
		self.b= np.array(np.random.uniform(low=-1, high=1, size=(length)))#output bias

	def sigmoid(self, x):
		aaa = math.exp(-x)
		return 1 / (1 + aaa)

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
		h=[]
		bina =[]
		for i in range(len(x)):
			nump = (np.matrix(self.w).transpose()[i])[:1, :len(x)]
			tot = self.b[i] + np.dot(nump, x).item(0,0) #bias plus the dot product
			tot = self.sigmoid(tot)
			h.append(tot)

		return h
		

	def back(self, h):
		x=[]
		bina =[]
		for i in range(len(h)):
			nump = (np.matrix(self.w)[i])[:1, :len(h)]
			tot = self.a[i] + np.dot(nump, h).item(0,0)
			tot = self.sigmoid(tot)
			x.append(tot)


		for i in range(len(h)):
			if x[i]>0.5:
				bina.append(1)
			else:
				bina.append(0)
		#if self.aq%1==0:
			#print x
			#print bina

		return x
		#return bina


'''tt = [1,0,0,0,0,0,0,0,0,0,0,0,0,1]
yy = [1,0,0,0,0,0,0,0,0,1,1,1,1,1]
bb= [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
p= RBM(len(tt))'''
'''
for i in range(100):
	p.aq+=1 #counter
	print "tt",
	p.process(tt)
	print "yy",
	p.process(yy)
	print "bb",
	p.process(bb)
'''

print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

p= RBM(784)
count = 0
val = 6
for i in range(len(mnist.train.images)/10):
	if not mnist.train.labels[i][val]:
		continue
	count+=1
	first_image = mnist.train.images[i]
	#print mnist.train.labels[i]
	print count, i
	p.process(mnist.train.images[i])
	'''plotData = mnist.train.images[i]
	print plotData
	for j in range(len(plotData)):
		if j%28==0:
			print ""
		print int(math.ceil(plotData[j])),

	print ""
	
	first_image = np.array(first_image, dtype='float')
	pixels = first_image.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.show()
	'''

for i in range(len(mnist.train.images)):
	if mnist.train.labels[i][val]:
		t = p.forward(mnist.train.images[i])
		plotData = p.back(t)

		'''
		for j in range(len(plotData)):
			if j%28==0:
				print ""
			if plotData[j] > 0.5:
				print 1,
			else:
				print 0,
			#print int(math.ceil(plotData[j])),

		print ""
		print ""
		print ""
		'''
		print time.clock()-beg
		first_image = plotData
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()

		first_image = mnist.train.images[i]
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()
		'''for j in range(len(plotData)):
			if j%28==0:
				print ""
			print plotData[j],
		'''
		break
