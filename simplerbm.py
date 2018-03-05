import numpy as np
import math

class RBM():
	learning_rate=1.1
	aq =0

	def __init__(self):
		self.w= np.random.uniform(low=-1, high=1, size=(8, 8))
		self.a= np.random.uniform(low=-1, high=1, size=(8))#input bias
		self.b= np.random.uniform(low=-1, high=1, size=(8))#output bias

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



	def forward(self, x):
		h=[]
		bina =[]
		for i in range(len(x)):
			nump = (np.matrix(self.w).transpose()[i])[:1, :len(x)]
			tot = self.b[i] + np.dot(nump, x).item(0,0)
			tot = self.sigmoid(tot)
			h.append(tot)

		for i in range(len(h)):
			if h[i]>0.5:
				bina.append(1)
			else:
				bina.append(0)
		#print h
		#print bina
		#recon = self.back(bina)
		recon = self.back(h)
		#print recon
		self.training(x, recon)
	
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
		if self.aq%100==0:
			print bina
		return x
		#print x
		#print bina


tt = [1,0,0]

p= RBM()
pp = np.random.uniform(low=-1, high=1, size=(8, 8))[:5,:5]
for i in range(1):
	p.aq+=1 #counter
	p.forward(tt)
	
	rr= p.w[:5, :5]
	
	if rr.all() != pp.all():
		print "yay"
	pp = rr


