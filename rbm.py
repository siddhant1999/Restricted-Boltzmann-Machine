import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
#for x in xrange(1,10):
batch = mnist.train.next_batch(True)
plotData = batch[0]
for i in range(len(plotData[0])):
	if i%28==0:
		print ""
	print int(math.ceil(plotData[0][i])),

print ""

#print len(plotData[0])
# plotData = plotData.reshape(28, 28)
# plt.gray() # use this line if you don't want to see it in color
# plt.imshow(plotData)
# plt.show()

class RBM():
	learning_rate=0.1
	

	def __init__(self):
		self.w= np.random.uniform(low=-1, high=1, size=(785,785))
		self.a= np.random.uniform(low=-1, high=1, size=(785))#input bias
		self.b= np.random.uniform(low=-1, high=1, size=(785))#output bias

		self.inp= [0 for j in range(785)]
		self.out= [0 for j in range(785)]
		#self.n[0][0]=1
		#print self.w
		#rint self.n

	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))
	def training(x):
		for i in range(len(x)):
			data=0
			for j in range(len(x)):
				s=self.b[j]
				for vi in range(len(x)):
					ss += x[vi]*self.w[i][j]
				data = sigmoid(ss)
	def feedforward(self, x):
		h=[]
		for i in range(len(x)):
			tot= self.b[i]
			for j in range(len(x)):
				tot += x[j]*self.w[j][i]
			h.append(tot)

a=[1,2,3]
asd=np.array([1,2,3])
print asd.transpose()
print asd
print np.ones((1, 2, 3))
b=[1,2,3]
print np.dot(b,a) 
x=RBM()
# x.feedforward()
