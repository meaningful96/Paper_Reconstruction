"""
Created on meaningful96

DL Project
"""
#%%
import torch
import numpy as np

#파이토치의 랜덤시드 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0) # gpu 1개 이상일 때 


np.random.seed(1)

a = torch.randn(1,2,3)
b = torch.randn(1,2,3)

c1 = torch.cat([a,b], dim = 0)
c2 = torch.cat([a,b], dim = 1)
c3 = torch.cat([a,b], dim = 2)
c4 = torch.cat([a,b], dim = -1)
c5 = torch.cat([a,b], dim = -2)
c6 = torch.cat([a,b], dim = -3)

print(a)
print(b)
print("----------------------------------------------")
print("c1")
print(c1)
print("----------------------------------------------")
print("c2")
print(c2)
print("----------------------------------------------")
print("c3")
print(c3)
print("----------------------------------------------")
print("c4")
print(c4)
print("----------------------------------------------")
print("c5")
print(c5)
print("----------------------------------------------")
print("c6")
print(c6)
print("----------------------------------------------")
print(c1 == c6)
print(c2 == c5)
print(c3 == c4)


#%%
import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

print(torch.stack([a, b], dim = 0))
'''
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]]), size = (2, 2, 3)'''

print(torch.stack([a, b], dim = 1))
'''
tensor([[[ 1,  2,  3],
         [ 7,  8,  9]],

        [[ 4,  5,  6],
         [10, 11, 12]]]), size = (2, 2, 3)'''

print(torch.stack([a, b], dim = 2))
'''
tensor([[[ 1,  7],
         [ 2,  8],
         [ 3,  9]],

        [[ 4, 10],
         [ 5, 11],
         [ 6, 12]]]), size = (2, 3, 2)'''


