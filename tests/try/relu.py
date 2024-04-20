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

tensor1 = Tensor([[0.1318359375, 0.458984375, 0.21875, 0.6796875],
[0.93359375, 0.51953125, 0.03466796875, 0.53125]])

tensor2 = Tensor([[0.0076904296875, 0.06689453125],
[0.6875, 0.9296875],
[0.52734375, 0.65234375],
[0.703125, 0.76171875]])

print(tensor1 @ tensor2)

[[0.9140625, 1.09375],
[0.7578125, 0.9765625]]