import torch

# Create a 1D tensor of ones with 3 elements
a = torch.ones(3)
print(a)
# tensor([1., 1., 1.])

a[1] = 0
# tensor([1., 0., 1.])

b = torch.zeros(3)
print(b)
# tensor([0., 0., 0.])

# Create a 3x2 tensor of zeros
points = torch.zeros(3, 2)
points
# tensor([[0., 0.],
#         [0., 0.],
#         [0., 0.]])


#torch also has datatypes
x = torch.ones(2, 2, dtype=torch.int16)
print(x)
# tensor([[1, 1],

#  torch.float32 or torch.float: 32-bit floating-point
#  torch.float64 or torch.double: 64-bit, double-precision floating-point
#  torch.float16 or torch.half: 16-bit, half-precision floating-point
#  torch.int8: signed 8-bit integers
#  torch.uint8: unsigned 8-bit integers
#  torch.int16 or torch.short: signed 16-bit integers
#  torch.int32 or torch.int: signed 32-bit integers
#  torch.int64 or torch.long: signed 64-bit integers
#  torch.bool: Boolean

#Transposing in higher dimensions
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
print(some_t.shape)
print(transpose_t.shape)

