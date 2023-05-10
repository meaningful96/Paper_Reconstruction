"""
Created on meaningful96

DL Project
"""
#%% torch tensor 곱하기
import torch
import torch.nn as nn

a = torch.randn(10)
b = torch.ones(4,10,5)
c = torch.matmul(a,b).size()
print(c) # torch.Size([4, 5])


d = torch.randn(5,4,1)
e = torch.ones(5,1,4)

f = torch.matmul(d,e).size()
print(f) # torch.Size([5, 4, 4])
         # 배치 차원이 같으니 나머지 차원끼지 곱셈이 진행됨.





# a = [1,2,3]
# b = tuple(a)
# c = {b}
# path_set = {1,3,5}
# path_list = list(path_set - c)

# path_list = list(path_set - {tuple([relation])})

#%% scipy lil 행렬
import scipy.sparse as sp

# create an empty 3x3 sparse matrix in LIL format
matrix = sp.lil_matrix((3, 3))

print(matrix)
# add some non-zero values to the matrix
matrix[0, 1] = 2
matrix[1, 2] = 3
matrix[2, 0] = 4

print(matrix.toarray())

s1 = set([1,2,3,4,5,6])
s2 = set([4,5,6,7,8,9])

r1 = s1 - s2
r2 = s2 - s1
r3 = s1.difference(s2)
r4 = s2.difference(s1)

print(r1) # {1,2,3}
print(r2) # {7,8,9}
print(r3) # {1,2,3}
print(r4) # {8,9,7}

#%% scipy issparse 함수
import numpy as np
from scipy import sparse

# create a sparse matrix using the COO format
data = np.array([1, 2, 3])
row = np.array([0, 1, 2])
col = np.array([0, 1, 2])
sparse_matrix = sparse.coo_matrix((data, (row, col)))

# check if the matrix is sparse
is_sparse = sparse.issparse(sparse_matrix)
print(is_sparse)  # Output: True

# create a dense matrix
dense_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# check if the matrix is sparse
is_sparse = sparse.issparse(dense_matrix)
print(is_sparse)  # Output: False

#%% Tensor 곱
import torch

t1 = torch.randn(3)
t2 = torch.randn(3)
out1 = torch.matmul(t1, t2) # size: 1

print(out1.size()) # torch.Size([]) 

#-------------------------------------------------#

t3 = torch.randn(3, 4)
t4 = torch.randn(4)
out2 = torch.matmul(t3, t4) # size: (3, )

print(out2.size()) # torch.Size([3])

#-------------------------------------------------#

t5 = torch.randn(10, 3, 4)
t6 = torch.randn(4)
out3 = torch.matmul(t5, t6) # size: (10, 3)

print(out3.size()) # torch.Size([10, 3])

#-------------------------------------------------#

t7 = torch.randn(10, 3, 4)
t8 = torch.randn(10, 4, 5)
out4 = torch.matmul(t7, t8) # size: (10, 3, 5)

print(out4.size()) # torch.Size([10, 3, 5])

#-------------------------------------------------#

x1 = torch.randn(10, 3, 4)
x2 = torch.randn(4, 5)
out5 = torch.matmul(x1, x2) # size: (10, 3, 5)

print(out5.size()) # torch.Size([10, 3, 5])
