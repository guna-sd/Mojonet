from net import Tensor, GELU

fn main():
    var x = Tensor[DType.float32](2,2)
    x[0] = 0.1315377950668335
    x[1] = 0.458650141954422
    x[2] = 1.21895918250083923
    x[3] = 0.67886471748352051
    print(x)
    var F = GELU[DType.float32]()
    print(F.forward(x))