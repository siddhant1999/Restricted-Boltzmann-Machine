import numpy as np
import math

class NeuralNet():
	inp = []
	outp = []
	learning_rate=0.1

	def __init__(self, inp, outp):
		self.inp = inp
		self.outp = outp
		self.w= np.random.uniform(low=-1, high=1, size=(3,3,3))
		self.n= [[0 for j in range(3)] for i in range(4)]
		#self.n[0][0]=1
		#print self.w
		#print self.n

	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))


	def forward(self, firstLayer):
		#set the neuron values
		for p in range(len(firstLayer)):
			self.n[0][p] = firstLayer[p]
		self.neurons=[2,3,3,1]

		#bias = np.random.uniform(low=-1, high=1, size=(3))
		self.bias_w = np.random.uniform(low=-1, high=1, size=(len(self.neurons)-1, max(self.neurons)))
		#bias[layer][destination_neuron]
		#this way you won't run into issues when increasing the size
		# the size of the bias array is one less than the number of layers
		#print "*"

		for i in range(1,len(self.neurons)):
			#i is layer number
			for j in range(self.neurons[i]):
				#j is the neuron in this layer
				#s= bias[i-1]
				s=self.bias_w[i-1][j]
				#s=0
				for k in range(self.neurons[i-1]):
					#k is the neuron in the previous layer
					s+= self.n[i-1][k] * self.w[i-1][k][j]

				self.n[i][j] = self.sigmoid(s)
		#print self.n[len(self.neurons)-1][0]
		#print "n: ", self.n
		'''print "-------------------------------------------"
		print self.w
		print "*******************************************"
		print self.n'''

	def outputLayer(self, ans):
		#in this case ans in the vector of outputs and must be of equal size to the last element of the neurons array
		totError = 0
		self.dup = self.w.copy() #this is a copy of the weights matrix so that we can update weights
		if self.neurons[-1] != len(ans):
			print "*error*"
			return
		deltaList=[]
		for i in range(self.neurons[-1]):#go through every neuron in the final layer
			target = ans[i]
			val = self.n[len(self.neurons)-1][i]
			totError += (target-val)*(target-val)/2.0
			#now that we have a neuron we are working with we should itterate over all the neurons
			delta = (val-target)*val*(1-val)
			deltaList.append(delta)
			for j in range(self.neurons[-2]):
				l = len(self.neurons)-2
				neuron = self.n[l][j] #weight from neuron j in layer -1 (last element) to the ith neuron of the next layer
				weight = self.w[l][j][i]
				self.dup[l][j][i] = weight-self.learning_rate * delta * neuron
			self.bias_w[len(self.neurons)-2][i] -= self.learning_rate * delta# don't * neuron becuase the neuron value of bias is always 1
				
		#this takes care of the first set
		#simply provide a list of the Eoi/doutoi which is just delta
		#print "Error:", totError
		#print np.round(totError*100, decimals=1)
		print np.round(totError, decimals=4)
		self.backprop(len(self.neurons)-2, deltaList)

	def backprop(self, layer, prev_deltas):

		if layer == 0:
			self.w = self.dup.copy()
			return

		newdeltaList = []
		for i in range(self.neurons[layer]):
			val = self.n[layer][i]
			deltasum = 0

			for j in range(self.neurons[layer+1]):
				if self.neurons[layer+1] != len(prev_deltas):
					print "OMG BAD"
					exit()
				weight = self.w[layer][i][j]
				eoi = prev_deltas[j]*weight
				deltasum+= eoi

			delta = deltasum*val*(1-val)
			newdeltaList.append(delta)
			#now itterate over every previous weight and modify it
			for k in range(self.neurons[layer-1]):
				neuron = self.n[layer-1][k]
				weight = self.w[layer-1][k][i]
				self.dup[layer-1][k][i] = weight - self.learning_rate * neuron * delta
			self.bias_w[layer-1][i] -= self.learning_rate*delta # I really hope this works
		self.backprop(layer-1, newdeltaList)





#a = [[1,0],[0,1],[1,1],[1,0],[0,0],[1,1],[0,1],[0,0]]
#b= [[1],[1],[0],[1],[0],[0],[1],[0]]

a = np.random.randint(low=0, high=2, size=(10000,2), dtype='l')
b=[]
for i in range(len(a)):
	if a[i][0] == a[i][1]:
		b.append([0])
	else:
		b.append([1])
# print a[1:100]
# print b[1:100]


x = NeuralNet(a, b)

for i in range(len(a)):
	#print a[i], b[i],
	x.forward(a[i])

	x.outputLayer(b[i])

#print x.bias_w

