from os.path import exists

fn mkdir( path: String, exists_ok : Bool) -> Bool:
    """
    Create a directory at the given path.
    """
    if not exists(path):
        if external_call["mkdir", Int, AnyPointer[Int8]](path._buffer.data) == 0:
            return True
        return False
    else:
        print("Directory already exists")
        return False

fn read_file(path : String) raises -> String:
    with open(path,'r') as file:
        return file.read()

fn write_file(content : String, path : String)raises:
    with open(path,'r') as file:
        file.write(content)
