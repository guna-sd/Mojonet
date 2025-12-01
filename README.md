<div align="center">
  <img src="assets/mojonet.png" alt="mojonet">
</div>

> > ⚠️ WARNING
> Mojonet is under development and the API is subject to change.


Mojonet is a ML/DL framework written in Mojo🔥.

### Latest Nightly
Use at your own risk and be patient!

---

## 🚧 Development Roadmap / TODO

### 🔹 Core Memory Infrastructure
- [x] **UniquePointer** — move-only smart pointer (ownership-safe)
- [x] **DataPointer** — device-aware smart pointer (deleter-managed)
- [x] **Allocator Registry & GlobalContext**
- [x] **Pluggable Allocator interface**
- [x] **Caching Allocators** (CPU / CUDA)  
- [x] **Memory Pools & Block Reuse**
- [x] **Storage & StorageImpl (ref-counted)**

### 🔹 Tensor System
- [ ] **Tensor API (basic constructors)**
- [ ] **Tensor metadata: shape, strides, dtype**
- [ ] **Layout system support (Strided first)**
- [ ] **Tensor views / slicing / reshape / expand**
- [ ] **Automatic broadcasting rules**

### 🔹 Device & Backend Abstraction
- [ ] **CPU backend**
- [ ] **CUDA backend**
- [ ] **Pluggable backend interface**
- [ ] **Optional integration with MAX (Modular Accelerated Compute)**

### 🔹 Kernels & Operations
- [ ] **Core math ops (add, mul, matmul, etc.)**
- [ ] **Reduction ops (sum, mean, …)**
- [ ] **Elementwise + universal functions**
- [ ] **BLAS integration (cuBLAS, etc.)**

### 🔹 Autograd & Training
- [ ] **Autograd engine**
- [ ] **Gradient tape & backward ops**
- [ ] **Optimizers (SGD, Adam, …)**

### 🔹 Distributed / Advanced
- [ ] **Multi-device support**
- [ ] **JIT / lazy execution**
- [ ] **Graph optimization**

---


> This project is experimental — contributions and feedback are welcome!  
> Stay tuned for updates as Mojonet evolves into a high-performance ML research stack 🌟
