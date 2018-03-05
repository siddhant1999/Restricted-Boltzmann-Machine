import numpy as np
import math

class RBM():
	learning_rate=0.01
	aq =0

	def __init__(self, length):
		self.w= np.array(np.random.uniform(low=-1, high=1, size=(length, length)))
		self.a= np.array(np.random.uniform(low=-1, high=1, size=(length)))#input bias
		self.b= np.array(np.random.uniform(low=-1, high=1, size=(length)))#output bias

		#self.inp= [0 for j in range(785)]
		#self.out= [0 for j in range(785)]
		#self.n[0][0]=1
		#print self.w

	def sigmoid(self, x):
		try:
			aaa = math.exp(-x)
		except OverflowError:
			#print "here"
			if x>0:
				return 1
			return -1
		return x

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
		h= self.forward(v)
		vPrime = self.back(h)
		hPrime = self.forward(vPrime)

		posGrad = np.array([v]).transpose()*np.array([h])
		negGrad = np.array([vPrime]).transpose()*np.array([hPrime])
		#print "a: ", self.a
		#print "big: ", self.a[:len(v)].transpose()
		self.w= np.array(self.w[:len(v), :len(v)]) + self.learning_rate*(posGrad-negGrad)
		#print self.w
		#self.a = np.array([self.a[:len(v)]]).transpose()- self.learning_rate*(np.array([v]).transpose()-np.array([vPrime]).transpose())
		#self.b = np.array([self.b[:len(v)]]).transpose()-self.learning_rate*(np.array([h]).transpose()-np.array([h]).transpose())
		#print "a: ", self.a
		#exit()
		#print self.w[:len(v), :len(v)]

	def forward(self, x):
		h=[]
		bina =[]
		for i in range(len(x)):
			nump = (np.matrix(self.w).transpose()[i])[:1, :len(x)]
			tot = self.b[i] + np.dot(nump, x).item(0,0)
			#print "b:", tot
			tot = self.sigmoid(tot)
			h.append(tot)

		for i in range(len(h)):
			if h[i]>0.5:
				bina.append(1)
			else:
				bina.append(0)

		return h
		#print h
		#print bina
		#recon = self.back(bina)
		
		#recon = self.back(h)

		#At this point we have the following
		#x is v
		#h is the hidden later (non-binary)
		#recon is v', the first construction


		#print recon
		#self.training(x, recon)
	
	def back(self, h):
		x=[]
		bina =[]
		for i in range(len(h)):
			nump = (np.matrix(self.w)[i])[:1, :len(h)]
			#print nump
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
		#print x
		#print bina


tt = [1,0,0,0,0,0,0,0,0,0,0,0,0,1]

p= RBM(len(tt))
pp = np.random.uniform(low=-1, high=1, size=(8, 8))[:5,:5]
for i in range(100):
	p.aq+=1 #counter
	p.process(tt)
	


