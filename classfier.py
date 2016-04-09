#coding=utf-8

from __future__ import division
import numpy as np
import math
import random
import matplotlib.pyplot as plt

class WeekClassfier:
	"""
		弱分类器
	"""
	def __init__(self):
		self.func = None

	def fit(self,x,y,weight = None):
		"""
			训练
		"""

		x = np.array(x)
		y = np.array(y)
		if weight is None:
			weight = np.ones(x.shape[0])
		w = np.zeros(x.shape[1])

		sum_weight = np.array(weight).sum()
		sum_err_rate_0 = reduce(lambda a,b: a + b[1] if b[0] == 1 else a,zip(y,weight),0)
		sum_err_rate_1 = reduce(lambda a,b: a + b[1] if b[0] != 1 else a,zip(y,weight),0)
		
		best_index = 0
		best = sum_weight
		best_fea = 0
		flag = False

		if sum_err_rate_0 < sum_err_rate_1:
			flag = False
			best = sum_err_rate_0
			best_index = x[:,best_fea].min() - 1
		else:
			flag = True
			best = sum_err_rate_1
			best_index = x[:,best_fea].min() - 1

		for i in xrange(x.shape[1]):
			tmp_x = x[:,i]
			l = sorted(zip(tmp_x,y,weight),key=lambda x:x[0])
			for j in l:
				if j[1] == 1:
					sum_err_rate_0 -= j[2]
					sum_err_rate_1 += j[2]
					if sum_err_rate_0 < best:
						best = sum_err_rate_0
						best_index = j[0]
						best_fea = i
						flag = False
				else:
					sum_err_rate_0 += j[2]
					sum_err_rate_1 -= j[2]
					if sum_err_rate_1 < best:
						best = sum_err_rate_1
						best_index = j[0]
						best_fea = i
						flag = True
		self.func = self.makeFun(best_index,best_fea,flag)
		return self.func

	def makeFun(self,x_bound,fea_index,flag):
		"""
			生成预测函数
		"""
		def fun(x):
			if x[fea_index] <= x_bound:
				return -1 if flag else 1
			else:
				return 1 if flag else -1

		return fun  
	def predict(self,x):
		"""
			预测
		"""
		return self.func(x)

class AdaBoost:
	def __init__(self,iters = 3):
		self.weakClassfier = WeekClassfier()
		self.funs = []
		self.iters = iters

	def fit(self,x,y):
		size = len(y)
		weight = np.ones(size) * (1/size)
		for i in xrange(self.iters):
			func = self.weakClassfier.fit(x,y,weight)
			err_rate = sum([c for a,b,c in zip(x,y,weight) if func(a) != b])/sum(weight) 
			alpha = 1/2 * math.log((1-err_rate)/err_rate)
			weight = map(lambda a :a[2] * math.exp(-alpha * a[1] *func(a[0])),zip(x,y,weight))
			weight = np.array(weight)/sum(weight)
			self.funs.append((func,alpha))

	def predict(self,x):
		ans = 0
		for func,alpha in self.funs:
			ans += func(x) * alpha

		return 1 if ans > 0 else -1


class Bagging:
	def __init__(self, sample_rate = 0.5, iters = 3):
		self.weakClassfier = WeekClassfier()
		self.funs = []
		self.iters = iters
		self.sample_rate = sample_rate

	def fit(self,x,y):
		for i in xrange(self.iters):
			samples = random.sample(zip(x,y),int(len(y) * self.sample_rate))
			train_x = np.array(map(lambda x: x[0],samples))
			train_y = np.array(map(lambda x: x[1],samples))
			func = self.weakClassfier.fit(train_x,train_y)
			self.funs.append(func)

	def predict(self,x):
		ans = 0
		for func in self.funs:
			ans += func(x)
		return 1 if ans > 0 else -1


if __name__ == '__main__':
	data = [(1,1,1),(1,3,1),(3,5,1),(4,6,1),(5,5,1),(2,1,-1),(2,4,-1),(4,3,-1),(6,2,-1),(6,6,-1)]
	data = np.array(data)
	x = data[:,:2]
	y = data[:,2]

	clf1 = AdaBoost(iters = 7)
	l1 = []
	clf1.fit(x,y)
	for i in xrange(len(y)):
		l1.append(clf1.predict(x[i]))
	
	clf2 = Bagging(sample_rate = 0.5, iters = 3000)
	l2 = []
	clf2.fit(x,y)
	for i in xrange(len(y)):
		l2.append(clf2.predict(x[i]))	

	clf3 = WeekClassfier()
	l3 = []
	clf3.fit(x,y)
	for i in xrange(len(y)):
		l3.append(clf3.predict(x[i]))
	print 'label'
	print y
	print 'adaboost'
	print l1
	print 'bagging'
	print l2
	print '弱分类器'
	print l3
	print 'adaboost 正确率'
	print sum([1 for a,b in zip(y,l1) if a == b])/len(y)
	print 'bagging 正确率'
	print sum([1 for a,b in zip(y,l2) if a == b])/len(y)
	print '弱分类器 正确率'
	print sum([1 for a,b in zip(y,l3) if a == b])/len(y)

	yl = {-1:'b',1:'r'}
	ll = {-1:'o',1:'x'}

	for i in xrange(len(y)):
		plt.scatter(x[i][0],x[i][1],marker = ll[l2[i]], color = yl[y[i]])

	plt.show()

