import numpy as np
import math

class RBM():
	learning_rate=0.1
	aq =0

	def __init__(self, length):
		self.w= np.array(np.random.uniform(low=-1, high=1, size=(length, length)))
		self.a= np.array(np.random.uniform(low=-1, high=1, size=(length)))#input bias
		self.b= np.array(np.random.uniform(low=-1, high=1, size=(length)))#output bias

	def sigmoid(self, x):
		aaa = math.exp(-x)
		return 1 / (1 + aaa)

	def training(self,v, h):

		for i in range(len(v)):
			for j in range(len(v)):
				s=self.b[j]
				for vi in range(len(v)):
					#print "hhehehehe", self.w[vi][j]
					s += v[vi]*self.w[vi][j]
				data = self.sigmoid(s)
				#self.b[j] -= self.learning_rate*(v[i]-data)
				#i might need to change this to j
				data*=v[i]

				t=self.a[j]

				for hj in range(len(h)):
					t += h[hj]*self.w[i][hj]
					#print h[hj], self.w[i][hj]

				#print t
				model = self.sigmoid(t)
				#self.a[i] -= self.learning_rate*(h[j]-data)

				model*=h[j]

				delta = self.learning_rate*(data-model)
				
				before = self.w[i][j]
				self.w[i][j] += delta
				#bprint self.w[i][j], before, delta
				#if delta != 0:
					#print delta, self.aq
				#print delta


	def process(self, v):
		h= self.forward(v) #this is the first hidden output
		vPrime = self.back(h) #reconstructed input
		hPrime = self.forward(vPrime) #reconstructed output

		vTran = np.array([v]).transpose()
		posGrad = vTran*np.array([h])
		vPrimeTran = np.array([vPrime]).transpose()
		negGrad = vPrimeTran*np.array([hPrime])

		self.w= np.array(self.w[:len(v), :len(v)]) + self.learning_rate*(posGrad-negGrad)
		

	def forward(self, x):
		h=[]
		bina =[]
		for i in range(len(x)):
			nump = (np.matrix(self.w).transpose()[i])[:1, :len(x)]
			tot = self.b[i] + np.dot(nump, x).item(0,0) #bias plus the dot product
			tot = self.sigmoid(tot)
			h.append(tot)

		'''
		for i in range(len(h)):
			if h[i]>0.5:
				bina.append(1)
			else:
				bina.append(0)
		#return bina
		'''
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
		if self.aq%1==0:
			#print x
			print bina

		return x
		#return bina


tt = [1,0,0,0,0,0,0,0,0,0,0,0,0,1]
yy = [1,0,0,0,0,0,0,0,0,1,1,1,1,1]
bb= [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
p= RBM(len(tt))
#pp = np.random.uniform(low=-1, high=1, size=(8, 8))[:5,:5]
for i in range(100):
	p.aq+=1 #counter
	print "tt",
	p.process(tt)
	print "yy",
	p.process(yy)
	print "bb",
	p.process(bb)
	


