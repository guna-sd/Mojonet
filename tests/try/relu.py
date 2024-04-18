from torch import Tensor
import torch

# def test_relu():
#     x = Tensor([[0.1315377950668335, 0.458650141954422],[1.21895918250083923, 0.67886471748352051]])
#     x.requires_grad = True
#     y = torch.nn.GELU().forward(x)
#     print(y.backward(x))

#     print(y[0][0].item())
#     print(y[0][1].item())
#     print(y[1][0].item())
#     print(y[1][1].item())
#     print(x==y)
# test_relu()
# 0.07265163213014603
# 0.3103947937488556
# 1.0831307172775269
# 0.5100910067558289

# class Module:

#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = 0

#     def parameters(self):
#         return []

inputs = Tensor([[0.1318359375, 0.458984375, 0.21875],
[0.6796875, 0.93359375, 0.51953125]],)

ten = torch.rand(2,3,2,4,5)
# print(ten.shape)
# tes = ten.transpose(-2,1)
# print(tes.shape)
print(ten.broadcast_to)