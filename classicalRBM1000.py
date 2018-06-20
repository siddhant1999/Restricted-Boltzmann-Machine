import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt


class RBM():
	#learning_rate=1
	aq =0

	def __init__(self, rows, cols, rate):
		self.w= np.array(np.random.uniform(low=-1, high=1, size=(rows, cols)))
		self.a= np.array(np.random.uniform(low=-1, high=1, size=(rows)))#input bias
		self.b= np.array(np.random.uniform(low=-1, high=1, size=(cols)))#output bias
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

#p= RBM(784, 5)
beg = time.time()
error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0
for ttt in range(100, 710, 10):
	for rrr in range(1, 11):
		rrrate=float(rrr)/20.0 
		p= RBM(794, ttt, rrrate)
		count = 0
		val = 5
		slow_down=550;
	
		print ttt, rrrate,
		#Training

		for i in range(len(mnist.train.images)/slow_down):
			s = mnist.train.labels[i]
	
			first_image = mnist.train.images[i]
			#if i%100==0:
				#print i
			#print count, i
			a=np.append(mnist.train.images[i], s)
			p.process(a)
			count+=1
		

		errtot = 0
		slow_down_again = 5
		correcte=0
		totale=0
		for i in range(len(mnist.validation.images)/slow_down_again):
			a= np.append(mnist.validation.images[i], [0,0,0,0,0,0,0,0,0,0])
			t= p.forward(a)
			plotData = p.back(t)
			predict = [round(x,2) for x in plotData[784:]]
			actual = mnist.validation.labels[i]
			diff= np.array(actual)-np.array(predict)
			error= np.dot(diff,diff)
			errtot += error

			if np.argmax(predict) == np.argmax(actual):
				correcte+=1
			totale+=1

		#print round(errtot,2),
		print correcte,

		if errtot<besterror:
			print "************",
			besterror = errtot
			bestrate = p.learning_rate
			bestnodes = ttt

		error_rates+=[errtot]
		correct=0
		total=0
		piv = 0.5
		slow_real_down = 10
	
		for i in range(len(mnist.test.images)/slow_real_down):
			a= np.append(mnist.test.images[i], [0,0,0,0,0,0,0,0,0,0])
			t= p.forward(a)
			plotData = p.back(t)
			predict = np.array(plotData[784:])
			actual = mnist.test.labels[i]
			if np.argmax(predict) == np.argmax(actual):
		
				correct+=1
			total+=1
		print("Correct: {}/{}".format(correct, total))
print "Best:"
print ("Run Time:", round(time.time()-beg, 2))
plt.show()
