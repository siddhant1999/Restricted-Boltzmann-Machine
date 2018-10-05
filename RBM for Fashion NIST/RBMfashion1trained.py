import json
import numpy as np
import math
import time
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxint)


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
error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0
for ttt in range(1):
	for rrr in range(1):
		#rrrate=float(rrr)/20.0
		#rrrate+=0.18 
		rrrate=0.1
		ttt=370
		p= RBM(794, ttt, rrrate)
		count = 0
		val = 5
		slow_down=1;
	
		print ttt, rrrate,
		#Training

		for i in range(len(mnist.train.images)/slow_down):
			s=np.zeros(10)
			s[mnist.train.labels[i]]= 1
	
			first_image = mnist.train.images[i]
			if i%100==0:
				print i
			#print count, i
			a=np.append(mnist.train.images[i], s)
			p.process(a)
			count+=1
		#Testing

		#t1=[]
		#e=[]
		errtot = 0
		print round(errtot,2)

		if errtot<besterror:
			besterror = errtot
			bestrate = p.learning_rate
			bestnodes = ttt
	
		error_rates+=[errtot]
		qw =p.w.tolist()
		w1=['w']
		equ=['=']
		a1=['a']
		b1=['b']
		qa =p.a.tolist()
		qb =p.b.tolist()

		with open("weights.json", 'wb') as outfile:
			outfile.write(json.dumps(','.join(w1)).replace('"', ''))
			outfile.write(json.dumps(','.join(equ)).replace('"', ''))
			json.dump(qw, outfile)
			outfile.write('\n')
			outfile.write(json.dumps(','.join(a1)).replace('"', ''))
			outfile.write(json.dumps(','.join(equ)).replace('"', ''))
			json.dump(qa, outfile)
			outfile.write('\n')
			outfile.write(json.dumps(','.join(b1)).replace('"', ''))
			outfile.write(json.dumps(','.join(equ)).replace('"', ''))
    			json.dump(qb, outfile)
		print p.w
		print p.a
		print p.b
		print " "
		print " "
		#plt.plot(t1, e, 'k')
print "Best:"
print besterror, bestnodes, bestrate

#print error_rates
print ("Run Time:", round(time.time()-beg, 2))
#plt.show()
