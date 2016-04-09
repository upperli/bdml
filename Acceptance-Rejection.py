#coding=utf-8
from __future__ import division
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def direct_sample(mu,sigma):
	"""
		直接采样
	"""
	#return 4 * np.random.randn()
	z = np.random.uniform(0,1)
	y = stats.norm.ppf(z, mu, sigma)
	return y

def plot(x,y1,y2,result):

	plt.title("Truncated normal distributing sampling")
	plt.xlabel('x')
	plt.ylabel('Density')

	plt.plot(x,y1,'r',label = 'true pdf')
	plt.plot(x,y2,'b',label = 'the shape of kq(z)')


	dic = dict()
	for r in result:
		dic.setdefault("%.1f"%r,0)
		dic["%.1f"%r] += 1

	#plt.hist(result,normed = True,histtype='step',alpha = 0.1,label = 'estimated pdf using the samples')
	items = map(lambda x: (float(x[0]), 10 * x[1]/len(result)) ,dic.items())


	ix = map(lambda x: x[0],items)
	iy = map(lambda x: x[1],items)
	plt.scatter(ix, iy,label = 'estimated pdf using the samples')

	plt.legend(loc = 1)
	plt.show()

def sample(k):
	while True:
		u = np.random.uniform(0,1)
		z = direct_sample(0,4)
		if u <= stats.norm.pdf(z,1,1)/(k * stats.norm.pdf(z,0,4))\
					and z >= 0 and z <=4:
			return z

if __name__ == '__main__':
	#采样数量
	N_samples = 10000
	
	x = np.linspace(0,4,1000)
	#截断正态分布 N(1,1)I(0<=x<=4)
	y1 = stats.norm.pdf(x,1,1) / \
		(stats.norm.cdf(4, 1, 1) - stats.norm.cdf(0, 1, 1))
	#N(0,4)
	y2 = stats.norm.pdf(x,0,4)
	#常量k
	k = (y1/y2).max()
	print k
	result = []
	for t in xrange(N_samples):
		result.append(sample(k))

	plot(x,y1,y2*k,result)
	