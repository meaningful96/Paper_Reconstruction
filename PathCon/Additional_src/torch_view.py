"""
Created on meaningful96

DL Project
"""

#%% view
import torch
import numpy as np

t = np.zeros((4,4,3)) # 0으로 채워진 4x4x3 numpy array 생성
ft = torch.FloatTensor(t) # 텐서로 변환
print(ft.shape)

a = ft.view([-1,3]) # 원래 4x4x3=48이고, [-1,3] 모양으로 변환: ?x3 = 48이다. 따라서 ? = 16
b = ft.view([2,-1]) # 원래 4x4x3=48이고, [2,-1] 모양으로 변환: 2x? = 48이다. 따라서 ? = 24
print(a.shape) # torch.Size([16, 3])
print(b.shape) # torch.Size([2, 24])

#%% reshape

import torch
import numpy as np

r = np.zeros((5, 5, 10))
fr = torch.FloatTensor(r)
print(fr.shape)

c = fr.reshape(10, 5, 5)
d = fr.reshape(1, -1)
print(c.shape) #torch.Size([10, 5, 5])
print(d.shape) #torch.Size([1, 250])