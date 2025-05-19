from memory import UnsafePointer
from mc10.memory.DataPointer import DataPointer
from testing import assert_equal, assert_true, assert_false
from sys import sizeof

# ===-------------------------------------------------------------------===#
# Test Helper Functions 
# ===-------------------------------------------------------------------===#

@always_inline
def create_test_memory(size: Int) -> DataPointer:
    """Create a test memory block for testing purposes."""
    var ptr: DataPointer
    ptr = DataPointer.alloc(size)
    return ptr

@always_inline
def setup_test_memory[T: DType](count: Int) -> DataPointer:
    """Set up a test memory block with sequential values."""
    var ptr: DataPointer
    ptr = DataPointer.alloc(count * sizeof[Scalar[T]]())
    
    # Initialize with sequential values
    for i in range(count):
        ptr.store(i, Scalar[T](i + 1))
        
    return ptr

# ===-------------------------------------------------------------------===#
# Basic Functionality Tests
# ===-------------------------------------------------------------------===#

def test_initialization():
    """Test various initialization methods."""
    # Default initialization (null pointer)
    var ptr1 = DataPointer()
    assert_equal(Int(ptr1), 0, "Default initialization should create null pointer")
    
    # Initialize from None
    var ptr2: DataPointer = None
    assert_equal(Int(ptr2), 0, "None initialization should create null pointer")
    
    # Copy from another DataPointer
    var ptr3 = create_test_memory(10)  # Initialize ptr3 for copying
    var ptr4 = ptr3.copy()
    assert_equal(Int(ptr4), Int(ptr3), "Copy should create identical pointer")

    # Initialize from a raw pointer
    var raw_ptr: DataPointer = DataPointer.alloc(10)
    var ptr5 = DataPointer(raw_ptr)
    assert_equal(Int(ptr5), Int(raw_ptr), "Raw pointer initialization should match raw pointer")

    # Initialize from a raw pointer
    var rawptr: UnsafePointer[Int] = UnsafePointer[Int].alloc(10)
    var ptr6 = DataPointer(rawptr)
    assert_equal(Int(ptr6), Int(rawptr), "Raw pointer initialization should match raw pointer")

    ptr1.free()
    ptr3.free()
    ptr4.free()
    ptr5.free()
    print("Initialization tests passed.")

def test_address_of():
    """Test address_of method."""
    var test_value: Int32 = 42
    var ptr = DataPointer.address_of(test_value)
    
    # Check that the pointer actually points to the test value
    assert_equal(ptr.load[DType.int32](), 42, "Address_of should return pointer to value")
    
    # Modify through pointer and verify
    ptr.store(Int32(99))
    assert_equal(test_value, 99, "Modification through pointer failed")

    print("Address_of test passed.")

def test_allocation():
    """Test memory allocation and deallocation."""
    var ptr: DataPointer
    var size = 128
    
    # Allocate memory
    ptr = DataPointer.alloc(size)
    assert_true(ptr != DataPointer(), "Allocation should return non-null pointer")
    
    # Verify memory is zero-initialized
    for i in range(size):
        assert_equal(ptr.load[Byte.element_type](i), Byte(0), "Memory should be zero-initialized")
    
    # Write and read back values
    for i in range(size):
        ptr.store(i, Byte(i % 256))
        
    for i in range(size):
        var value = ptr.load[Byte.element_type](i)
        assert_equal(value, Byte(i % 256), "Memory read/write failed")
    
    # Free memory
    ptr.free()
    print("Memory allocation test passed.")
# ===-------------------------------------------------------------------===#
# Operator Tests
# ===-------------------------------------------------------------------===#

def test_pointer_arithmetic():
    """Test pointer arithmetic operations."""
    var base_ptr = create_test_memory(100)
    
    # Test addition
    var ptr1 = base_ptr + 10
    assert_equal(Int(ptr1), Int(base_ptr) + 10, "Pointer addition failed")
    
    # Test subtraction
    var ptr2 = ptr1 - 5
    assert_equal(Int(ptr2), Int(ptr1) - 5, "Pointer subtraction failed")
    
    # Test in-place addition
    var ptr3 = base_ptr.copy()
    ptr3 += 20
    assert_equal(Int(ptr3), Int(base_ptr) + 20, "In-place addition failed")
    
    # Test in-place subtraction
    ptr3 -= 10
    assert_equal(Int(ptr3), Int(base_ptr) + 10, "In-place subtraction failed")
    
    # Clean up
    base_ptr.free()
    print("Pointer arithmetic tests passed.")

def test_comparison_operators():
    """Test pointer comparison operators."""
    var ptr1 = create_test_memory(10)
    var ptr2 = ptr1 + 5
    
    # Test equality
    assert_true(ptr1 == ptr1, "Same pointer should be equal")
    assert_false(ptr1 == ptr2, "Different pointers should not be equal")
    
    # Test inequality
    assert_true(ptr1 != ptr2, "Different pointers should be not equal")
    assert_false(ptr1 != ptr1, "Same pointer should not be not equal")
    
    # Test less than
    assert_true(ptr1 < ptr2, "Lower address should be less than higher address")
    assert_false(ptr2 < ptr1, "Higher address should not be less than lower address")
    
    # Test less than or equal
    assert_true(ptr1 <= ptr2, "Lower address should be less than or equal to higher address")
    assert_true(ptr1 <= ptr1, "Same pointer should be less than or equal to itself")
    assert_false(ptr2 <= ptr1, "Higher address should not be less than or equal to lower address")
    
    # Test greater than
    assert_true(ptr2 > ptr1, "Higher address should be greater than lower address")
    assert_false(ptr1 > ptr2, "Lower address should not be greater than higher address")
    
    # Test greater than or equal
    assert_true(ptr2 >= ptr1, "Higher address should be greater than or equal to lower address")
    assert_true(ptr1 >= ptr1, "Same pointer should be greater than or equal to itself")
    assert_false(ptr1 >= ptr2, "Lower address should not be greater than or equal to higher address")
    
    # Clean up
    ptr1.free()

    print("Pointer comparison tests passed.")

def test_boolean_conversion():
    """Test conversion to boolean."""
    var null_ptr = DataPointer()
    var valid_ptr = create_test_memory(10)
    
    # Test __bool__ and __as_bool__
    assert_false(null_ptr, "Null pointer should convert to False")
    assert_true(valid_ptr, "Valid pointer should convert to True")
    
    # Clean up
    valid_ptr.free()
    print("Boolean conversion tests passed.")

def test_integer_conversion():
    """Test conversion to integer."""
    var ptr = create_test_memory(10)
    var addr = Int(ptr)
    
    # Test __int__ and __as_int__
    assert_equal(Int(ptr), addr, "__int__ method failed")
    assert_equal(ptr.__as_int__(), addr, "__as_int__ method failed")
    
    # Clean up
    ptr.free()
    print("Integer conversion tests passed.")

# ===-------------------------------------------------------------------===#
# Memory Alignment Tests
# ===-------------------------------------------------------------------===#

def test_is_aligned():
    """Test alignment checking."""
    # Create an unaligned pointer by allocating and offsetting by 1
    var base_ptr = create_test_memory(100)
    var unaligned_ptr = base_ptr + 1
    
    # Test different alignment values
    for alignment in List(1, 2, 4, 8, 16, 32, 64):
        # Check base pointer (should be at least 8-byte aligned by default)
        if alignment[] <= 8:
            assert_true(base_ptr.is_aligned(alignment[]), 
                        String("Base pointer should be aligned to ", alignment[]))
        
        # Check unaligned pointer
        if alignment[] > 1:
            assert_false(unaligned_ptr.is_aligned(alignment[]), 
                        String("Unaligned pointer should not be aligned to ", alignment[]))
        else:
            assert_true(unaligned_ptr.is_aligned(alignment[]), 
                        "Any pointer should be aligned to 1-byte boundary")
    
    # Clean up
    base_ptr.free()
    print("Alignment tests passed.")

def test_align_up():
    """Test align_up method."""
    var base_ptr = create_test_memory(100)
    
    # Test alignment for various offsets and alignments
    for offset in List(1, 3, 7, 13):
        var unaligned_ptr = base_ptr + offset[]
        
        for alignment in List(2, 4, 8, 16):
            var aligned_ptr = unaligned_ptr.align_up(alignment[])
            
            # Check the aligned pointer is actually aligned
            assert_true(aligned_ptr.is_aligned(alignment[]), 
                        String("align_up should create pointer aligned to ", alignment[]))
            
            # Check the aligned pointer is >= the original pointer
            assert_true(aligned_ptr >= unaligned_ptr, 
                        "align_up should round up to higher or equal address")
            
            # Check that the difference is minimal (less than alignment)
            assert_true(Int(aligned_ptr) - Int(unaligned_ptr) < alignment[],
                        "align_up should pick closest aligned address")
    
    # Clean up
    base_ptr.free()
    print("Align up tests passed.")

def test_align_down():
    """Test align_down method."""
    var base_ptr = create_test_memory(100)
    
    # Test alignment for various offsets and alignments
    for offset in List(1, 3, 7, 13):
        var unaligned_ptr = base_ptr + offset[]
        
        for alignment in List(2, 4, 8, 16):
            var aligned_ptr = unaligned_ptr.align_down(alignment[])
            
            # Check the aligned pointer is actually aligned
            assert_true(aligned_ptr.is_aligned(alignment[]), 
                        String("align_down should create pointer aligned to ", alignment[]))
            
            # Check the aligned pointer is <= the original pointer
            assert_true(aligned_ptr <= unaligned_ptr, 
                        "align_down should round down to lower or equal address")
            
            # Check that the difference is minimal (less than alignment)
            assert_true(Int(unaligned_ptr) - Int(aligned_ptr) < alignment[],
                        "align_down should pick closest aligned address")
    
    # Clean up
    base_ptr.free()
    print("Align down tests passed.")

# ===-------------------------------------------------------------------===#
# Memory Operations Tests
# ===-------------------------------------------------------------------===#

def test_scalar_load_store():
    """Test scalar load and store operations."""
    var ptr = create_test_memory(100)
    
    # Test different scalar types
    # Int32
    ptr.store(Int32(12345))
    assert_equal(ptr.load[Int32.element_type](), Int32(12345), "Int32 store/load failed")
    
    # Float32
    ptr.store(Float32(3.14159))
    assert_equal(ptr.load[Float32.element_type](), Float32(3.14159), "Float32 store/load failed")
    
    # Test with offsets
    ptr.store(5, Int32(54321))
    assert_equal(ptr.load[Int32.element_type](5), Int32(54321), "Int32 store/load with offset failed")
    
    ptr.store(10, Float32(2.71828))
    assert_equal(ptr.load[Float32.element_type](10), Float32(2.71828), "Float32 store/load with offset failed")
    
    # Clean up
    ptr.free()
    print("Scalar load/store tests passed.")

def test_simd_load_store():
    """Test SIMD vector load and store operations."""
    # Skip test if SIMD is not available
    alias simd_size = 4  # 4 element SIMD vector
    
    var ptr = create_test_memory(100)
    
    # Ensure pointer is aligned for SIMD operations
    var aligned_ptr = ptr.align_up(16)
    
    # Test with Float32 SIMD
    var simd_value = SIMD[Float32.element_type, simd_size](1.0, 2.0, 3.0, 4.0)
    aligned_ptr.store(simd_value)
    
    var loaded_value = aligned_ptr.load[Float32.element_type, simd_size]()
    for i in range(simd_size):
        assert_equal(loaded_value[i], simd_value[i], "SIMD store/load failed")
    
    # Test with offset
    var offset_simd = SIMD[Float32.element_type, simd_size](5.0, 6.0, 7.0, 8.0)
    aligned_ptr.store(4, offset_simd)  # Store at offset 4 elements
    
    var loaded_offset = aligned_ptr.load[Float32.element_type, simd_size](4)
    for i in range(simd_size):
        assert_equal(loaded_offset[i], offset_simd[i], "SIMD store/load with offset failed")
    
    # Clean up
    ptr.free()
    print("SIMD load/store tests passed.")

def test_memset():
    """Test memset operation."""
    var size = 100
    var ptr = create_test_memory(size)
    
    # Fill memory with a value
    var fill_value = Byte(42)
    ptr.memset(fill_value, size)
    
    # Verify all bytes are set correctly
    for i in range(size):
        assert_equal(ptr.load[Byte.element_type](i), fill_value, "memset failed")
    
    # Test partial memset
    var partial_start = 25
    var partial_size = 50
    var partial_value = Byte(99)
    
    t = (ptr + partial_start)
    t.memset(partial_value, partial_size)
    
    # Verify bytes before partial section remain unchanged
    for i in range(partial_start):
        assert_equal(ptr.load[Byte.element_type](i), fill_value, "memset modified bytes outside range")
    
    # Verify bytes in partial section are changed
    for i in range(partial_start, partial_start + partial_size):
        assert_equal(ptr.load[Byte.element_type](i), partial_value, "partial memset failed")
    
    # Verify bytes after partial section remain unchanged
    for i in range(partial_start + partial_size, size):
        assert_equal(ptr.load[Byte.element_type](i), fill_value, "memset modified bytes outside range")
    
    # Clean up
    ptr.free()
    print("memset tests passed.")

def test_copy_from():
    """Test memory copying."""
    var src_size = 50
    var dest_size = 100
    
    var src_ptr = create_test_memory(src_size)
    var dest_ptr = create_test_memory(dest_size)
    
    # Initialize source memory with pattern
    for i in range(src_size):
        src_ptr.store(i, Byte((i * 3) % 256))
    
    # Copy entire source to destination
    dest_ptr.copy_from(src_ptr, src_size)
    
    # Verify copy worked correctly
    for i in range(src_size):
        assert_equal(dest_ptr.load[Byte.element_type](i), src_ptr.load[Byte.element_type](i), "copy_from failed")
    
    # Test partial copy
    var src_offset = 10
    var dest_offset = 20
    var copy_size = 25
    
    t = (dest_ptr + dest_offset)
    t.copy_from(src_ptr + src_offset, copy_size)
    
    # Verify partial copy
    for i in range(copy_size):
        assert_equal(
            dest_ptr.load[Byte.element_type](dest_offset + i),
            src_ptr.load[Byte.element_type](src_offset + i),
            "partial copy_from failed"
        )
    
    # Clean up
    src_ptr.free()
    dest_ptr.free()
    print("copy_from tests passed.")

# ===-------------------------------------------------------------------===#
# Type Casting Tests
# ===-------------------------------------------------------------------===#

def test_address_space_cast():
    """Test address space casting."""
    var ptr = create_test_memory(10)
    
    # Cast to different address space (exact test depends on implementation)
    var generic_ptr = ptr.address_space_cast[AddressSpace.GENERIC]()
    
    # Verify the cast preserves the address
    assert_equal(Int(generic_ptr), Int(ptr), "Address space cast changed address")
    
    # Clean up
    ptr.free()
    generic_ptr.free()
    print("Address space cast test passed.")

def test_unsafe_ptr():
    """Test unsafe_ptr method."""
    var ptr = create_test_memory(10)
    
    # Get unsafe pointer
    var unsafe = ptr.unsafe_ptr()
    
    # Verify the address is preserved
    assert_equal(Int(__type_of(unsafe)(unsafe)), Int(ptr), "unsafe_ptr changed address")
    
    print("Unsafe pointer test passed.")

# ===-------------------------------------------------------------------===#
# Edge Case Tests
# ===-------------------------------------------------------------------===#

def test_zero_length_operations():
    """Test operations with zero length."""
    var ptr = create_test_memory(10)
    
    # Initialize with known value
    ptr.memset(Byte(42), 10)
    
    # Zero-length memset should do nothing
    ptr.memset(Byte(0), 0)
    assert_equal(ptr.load[Byte.element_type](0), Byte(42), "Zero-length memset modified memory")
    
    # Zero-length copy should do nothing
    var other_ptr = create_test_memory(10)
    other_ptr.memset(Byte(99), 10)
    ptr.copy_from(other_ptr, 0)
    assert_equal(ptr.load[Byte.element_type](0), Byte(42), "Zero-length copy modified memory")
    
    # Clean up
    ptr.free()
    other_ptr.free()
    print("Zero-length operations test passed.")

def test_null_pointer_checks():
    """Test null pointer behavior."""
    var null_ptr = DataPointer()
    
    # Check null pointer detection
    assert_false(null_ptr, "Null pointer should evaluate to false")
    assert_equal(Int(null_ptr), 0, "Null pointer should have address 0")
    
    # Check null pointer is aligned to any alignment
    for alignment in List(1, 2, 4, 8, 16):
        assert_true(null_ptr.is_aligned(alignment[]), "Null pointer should be aligned to any boundary")
    
    print("Null pointer checks passed.")

# ===-------------------------------------------------------------------===#
# Run all tests
# ===-------------------------------------------------------------------===#

def test_ptr():
    # Run all test cases
    test_initialization()
    test_address_of()
    test_allocation()
    test_pointer_arithmetic()
    test_comparison_operators()
    test_boolean_conversion()
    test_integer_conversion()
    test_is_aligned()
    test_align_up()
    test_align_down()
    test_scalar_load_store()
    test_simd_load_store()
    test_memset()
    test_copy_from()
    test_address_space_cast()
    test_unsafe_ptr()
    test_zero_length_operations()
    test_null_pointer_checks()