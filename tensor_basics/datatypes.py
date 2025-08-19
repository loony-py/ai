import torch

float_32_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=None)
print(f"Float 32 tensor: {float_32_tensor}")
print(f"Float 32 tensor datatype: {float_32_tensor.dtype}")
print("\n")

float_16_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
print(f"Float 16 tensor: {float_16_tensor}")
print(f"Float 16 tensor datatype: {float_16_tensor.dtype}")
print("\n")

xx = float_32_tensor * float_16_tensor
print(f"Mutiply float 32 and float 16. Datatype: {xx.dtype}")

int_32 = torch.tensor([1, 2, 3], dtype=torch.int32)
print(f"Int 32 tensor: {int_32}")
print(f"Int 32 tensor datatype: {int_32.dtype}")
print(f"\n")

yy = float_32_tensor * int_32
print(f"Mutiply float 32 and int 32. Datatype: {yy.dtype}")
