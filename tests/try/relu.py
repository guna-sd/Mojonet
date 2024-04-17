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

inputs = Tensor([[0.1315377950668335, 0.458650141954422],[0.21895918250083923, 0.67886471748352051]])
print(torch.sigmoid(inputs))
