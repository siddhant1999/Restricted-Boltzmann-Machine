from random import randint
import numpy as np
import math
import time
import sys
import tensorflow as tf
import para
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxint)

labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker' ,'Bag', 'Ankle Boot']

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


print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

beg = time.time()
weight_matrix = np.array(para.w)
p =RBM(794, 200, 0.1, weight_matrix, np.array(para.a), np.array(para.b))

error_rates =[]
besterror = 9999
bestrate = 0
bestnodes = 0

slow_down=1

correct=0
total=0
piv = 0.1
#im =np.random.shuffle(np.array(mnist.test.images))
for j in range(len(mnist.test.images)/slow_down):
	i = randint(0, len(mnist.test.images)/slow_downc)
	a= np.append(mnist.test.images[i], np.zeros(10))
	t= p.forward(a)
	plotData = p.back(t)
	predict = plotData[784:]
	actual = mnist.test.labels[i]
	#print i
	
	fi = plt.figure(figsize=(8,8))
	
	first_image = plotData[:784]
	first_image = np.array(first_image, dtype='float')
	pixels = first_image.reshape((28, 28))
	fi.add_subplot(2,2, 2)
	plt.title(labels[np.argmax(predict)], fontsize=15)
	plt.xlabel('Prediction')
	plt.imshow(pixels, cmap='gray')
	
	
	sec =np.array(mnist.test.images[i], dtype='float')
	pix = sec.reshape((28, 28))
	fi.add_subplot(2,2, 1)
	plt.title(labels[actual], fontsize=15)
	plt.xlabel('Actual')
	plt.imshow(pix, cmap='gray')

	fi.add_subplot(2,2,3)
	zz=[0,1,2,3,4,5,6,7,8,9]
	if np.argmax(predict) == actual:
		plt.bar(zz, plotData[784:], color='green')
	else:
		plt.bar(zz, plotData[784:], color='red')
	for oo, txt in enumerate(plotData[784:]):
		plt.annotate(int(txt*100), (zz[oo], plotData[784:][oo]))
	plt.ylim([0,1])

	fi.add_subplot(2,2,4)
	
	dat = plotData[784:]
	dat = np.argsort(dat)
	dat = dat[::-1]
	strr = ""
	indexc = 1
	for ts in dat:
		strr+= str(indexc)+ ". "
		strr+=labels[ts]
		strr+="\n"
		indexc+=1
	confid = "Confidence: " + str(int(np.amax(plotData[784:])/np.sum(plotData[784:])*100)) + "%"
	if np.argmax(predict) == actual:
		plt.text(0.3,0.93,confid,fontsize=15, color='green')
	else:
		plt.text(0.3,0.92,confid,fontsize=15, color='red')
	plt.text(0.3,0,strr,fontsize=15)
	

	plt.axis('off')

	plt.show()
	
	if np.argmax(predict) == actual:
		correct+=1
		
	total+=1

print float(correct)/float(total)
plt.plot(piv, float(correct)/float(total), '-o')
#e+=[error]
#errtot += error
#add a breakdown of which numbers are problemsome
print "Correct: ", correct, "/", total, "Pivot:", piv
print ("Run Time:", round(time.time()-beg, 2))
#plt.show()
