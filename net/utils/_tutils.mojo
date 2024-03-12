from collections.vector import DynamicVector
from math import exp
from random import random_float64
from tensor import Tensor



fn map(vector : DynamicVector[Float64], f : fn(Float64)->Float64) -> DynamicVector[Float64]:
    var new = DynamicVector[Float64]()
    for i in range(vector.size):
        new.push_back(f(vector[i]))
    return new
    
fn reverse_vector_float(vector : DynamicVector[Float64]) -> DynamicVector[Float64]:
    var reverse_vector = DynamicVector[Float64]()
    for i in range(vector.size -1, -1, -1):
        reverse_vector.push_back(vector[i])
    return reverse_vector

fn reverse_vector_int(vector : DynamicVector[Int]) -> DynamicVector[Int]:
    var reverse_vector = DynamicVector[Int]()
    for i in range(vector.size -1, -1, -1):
        reverse_vector.push_back(vector[i])
    return reverse_vector

fn print_vector_float(vector : DynamicVector[Float64]):
    var s = String()
    s += "["
    for i in range(vector.size):
        s += vector[i]
        s += " "
    s += "]"

    print(s)

fn print_vector_int(vector : DynamicVector[Int]):
    var s = String()
    s+= "["
    for i in range(vector.size):
        s+=vector[i]
        s+= " "
    s+="]"

    print(s)
