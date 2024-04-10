from memory.unsafe import DTypePointer



@value
struct Stack[T : DType]:
    var top : Int
    var size : Int
    var data: DTypePointer[T]
        
    fn __init__(inout self, Size: Int):
        self.top = -1
        self.size = Size
        self.data = DTypePointer[T]().alloc(self.size)
 
    fn isFull(inout self) -> Bool:
        return self.top == self.size
    
    fn isEmpty(inout self) -> Bool:
        return self.top == -1
    
    fn push(inout self,value : SIMD[T,1]):
        if self.isFull():
            print('Stack is full')
            return
        self.top += 1
        self.size += 1
        self.data.store(self.top, value)
               
    fn pop(inout self):
        if self.top == -1:
            print('Stack is empty')
        self.top = self.top - 1
    
    fn peek(inout self) -> SIMD[T,1]:
        if self.top == -1:
            print('Stack is empty')
        return self.data[self.top]

    fn clear(inout self):
        self.top = -1
        self.size = 0
        self.data.free()

    fn __str__(self: Self) -> String:
        var buf = String()
        for i in range(self.size):
            buf += str(self.data[i])
            buf += "\n"
        return buf

    fn __del__(owned self):
        self.data.free()


fn main():
    var stack = Stack[DType.int32](10)
    stack.push(6)
    stack.push(7)
    print(stack)