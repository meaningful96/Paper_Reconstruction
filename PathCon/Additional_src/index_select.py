"""
Created on meaningful96

DL Project
"""

import torch
x = torch.randn(3, 4)

in1 = torch.tensor([1,2])
in2 = torch.tensor([0,2])
# in3 = torch.tensor([1,1,3])

# in4 = torch.tensor([0,1,3])

a1 = torch.index_select(x, dim = 0, index = in1)
a2 = torch.index_select(x, dim = 1, index = in1)
# b1 = torch.index_select(x, dim = 0, index = in2)
# b2 = torch.index_select(x, dim = 1, index = in2)
# # c1 = torch.index_select(x, dim = 0, index = in3) 
# # c2 = torch.index_select(x, dim = 1, index = in3)
# # d1 = torch.index_select(x, dim = 0, index = in4)
# d2 = torch.index_select(x, dim = 1, index = in4)
print(x)
print(a1)
print(a2)
print()

b1 = torch.index_select(x, dim = 0, index = in2)
b2 = torch.index_select(x, dim = 1, index = in2)

print(x)
print(b1)
print(b2)

# dim = 0은 가로방향으로, index = torch.tensor([1,2])일 경우 1,2번 줄 즉 즉. 1행 2행 출력
# "                                          [0,2] 일 경우 0, 2번 줄  0행 1행
# dim = 1은 세로방향으로, index = torch.tensor([1,2])일 경우 1,2번 줄 즉 즉. 1열 2열 
# "                                          [0,2] 일 경우 0, 2번 줄  0열 2열

