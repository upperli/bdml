#coding=utf-8
import numpy as np
import math


def fun(x):
	"""
		被积分函数
	"""
	return math.atan(x)/(x*x + x * math.sin(x))

N_samples = 1000
F_N = []

Z_N = np.random.uniform(0,1,N_samples)

for i in xrange(N_samples):
	F_N.append(fun(Z_N[i]))

mean = np.mean(F_N)
var  = np.var(F_N)
std = np.std(F_N)

print "mean: %f"%mean
print "var: %f"%var
print "std: %f"%std

