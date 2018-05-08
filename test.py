import numpy as np
file=open("testing123.txt","r")
a=file.read().split('[')
b=a[-2].split(' ')#maybe should be a space
#print b

final =[]
for j in range(1, len(a)):
	b=a[j].split(',')
	temp=[]
	for i in range(len(b)):
		b[i] =b[i].strip()
		if len(b[i]) < 1:
			continue
		if b[i][-1] == ']':
			b[i]=b[i][0:-1]
		
		print "::", b[i], "::"
		temp.append(float(b[i]))
	final.append(temp)
		#print float(b[i])

print final
print np.shape(np.array(final))
#print b
