import numpy as np
from numpy import genfromtxt
import json
import math
import time
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=sys.maxint)

my_data = genfromtxt('lung.csv', delimiter=',')
#print np.amax(np.array(my_data))
my_data = np.divide(np.array(my_data), 15)

b = np.zeros((104,256))
for i in range(54):
	b[i][-1]= 1
b[:,:-1] = my_data
my_data = b
#print my_data
np.random.shuffle((my_data))

train = my_data[:80]
validation = my_data[80:90]
test = my_data[90:]
#print train
#print validation
#print test


'''
print train
print ""
print validation
print ""
print test
print np.shape(train)
print np.shape(validation)
print np.shape(test)

'''
piv = 0.5

def pivret(x):
	if x > piv:
		return 1
	return 0

	

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

beg = time.time()
error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0




for ttt in range(1,250,5):
	for rrr in range(1, 20):
		rrrate=float(rrr)/20.0
		
		p= RBM(256, ttt, rrrate)
		count = 0
		val = 5
		slow_down=1;
	
		print ttt, rrrate,
		#Training

		for i in range(len(train)/slow_down):
			p.process(train[i])
		
		errtot = 0
		for i in range(len(validation)/slow_down):
			
			a = np.array(validation[i])
			a[-1] = 0
			t= p.forward(a)
			plotData = p.back(t)
			predict = plotData[255]
			actual = validation[i][-1]
			#print "Prediction:", predict, actual
			error= (actual - predict) *(actual - predict)
			
			errtot += error
		print round(errtot,2),

		if errtot<besterror:
			print "***",
			
			correct=0
			total=0
			for j in range(len(test)/slow_down):
				#print test[j][255:], test[j][-1],
				#a= np.append(test[j], [0])#perhaps we should try to use 0.5 instead of 0 or as the clamp
				a=np.array(test[j])
				a[-1]=0
				t= p.forward(a)
				plotData = p.back(t)
				predict = plotData[255]
				actual = test[j][-1]

				#print "Prediction:", predict, actual
				if pivret(predict) ==  actual:
					correct+=1
				total+=1
			print "Correct: ", correct
			#exit()



			besterror = errtot
			bestrate = p.learning_rate
			bestnodes = ttt
		else:
			print ""
		error_rates+=[errtot]
		
		#plt.plot(t1, e, 'k')
print "Best:"
print besterror, bestnodes, bestrate

#print error_rates
print ("Run Time:", round(time.time()-beg, 2))
#plt.show()
