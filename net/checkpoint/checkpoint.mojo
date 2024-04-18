from .fileutils import read_file, write_file


struct FileBuffer:
  var data: DTypePointer[DType.uint8]
  var offset: Int
  var size: Int

  fn __init__(inout self):
    self.data = DTypePointer[DType.uint8]()
    self.offset = 0
    self.size = 0

  fn __del__(owned self):
    self.data.free()
