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

tensor1 = Tensor([[-32768, -24148, 16752, -2709],
[2148, -18418, -29685, 11723]]).type(torch.int16)

tensor2 = Tensor([[11751, 28489, -7635, 1273]]).type(torch.int16)

print(tensor1.add(tensor2))

[[0.9140625, 1.09375],
[0.7578125, 0.9765625]]

[[-21017.,   4341.,   9117.,  -1436.],
        [ 13899.,  10071., -37320.,  12996.]]
tensor([[-21017,   4341,   9117,  -1436],
        [ 13899,  10071,  28216,  12996]], dtype=torch.int16)