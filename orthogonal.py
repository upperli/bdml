#coding=utf-8

import numpy as np

#同一个面上所有点的排列可能，一个面有三列，三个数字分别表示三列上点的位置
face = []
for i in xrange(3):
	for j in xrange(3):
		if i == j:
			continue
		for k in xrange(3):
			if i == k or j == k:
				continue
			face.append((i,j,k))

#计数
count = 0

#立方体总共有三层，枚举所有可能
for i in xrange(len(face)):
	for j in xrange(len(face)):
		if ((np.array(face[i]) - np.array(face[j])) == 0).any():
			continue
		for k in xrange(len(face)):
			if ((np.array(face[i]) - np.array(face[k])) == 0).any():
				continue
			if ((np.array(face[j]) - np.array(face[k])) == 0).any():
				continue
			print np.array(face[i])
			print np.array(face[j])
			print np.array(face[k])
			print '---------------'
			count += 1

#可能的数量
print count

