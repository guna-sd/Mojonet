from memory.unsafe import Pointer


struct Stack[T : AnyType]:
    var top : Int
    var size : Int
    var data: Pointer[T]
        
    fn __init__(inout self, Size: Int):
        self.top = 0-1
        self.size = Size
        self.data = Pointer[T].alloc(self.size)
        
    fn isFull(inout self) -> Bool:
        return self.top == self.size
    
    fn push(inout self,value : T):
        if self.isFull():
            print('Stack is full')
            return
        self.top = self.top + 1
        self.data.store(self.top, value)
               
    fn pop(inout self):
        if self.top == -1:
            print('Stack is empty')
        self.top = self.top - 1

    # fn print_stack(inout self):
    #     for i in range(self.size):
    #         print(self.data[i])

    fn __del__(owned self):
        self.data.free()


fn main():
    var stack = Stack[Int](10)

    stack.push(5)
    stack.push(10)
    stack.push(11)