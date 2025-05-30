import math
import torch
from torch import Tensor
from typing import Optional, List
from quantized_training.mx_utils import _reshape_to_blocks, _undo_reshape_to_blocks
from quantized_training.decomposed import vmap
from quantized_training.fake_quantize import get_quantization_map

INT8_MAP = get_quantization_map("int8")

def simple_mx_quantize(
    X: Tensor,
    code: Tensor = INT8_MAP,
    quant_max: float = 127.0,
    block_size: int = 32,
    axes: Optional[List[int]] = None,
    shared_exp_method: str = "max",
    force_scale_power_of_two: bool = True,
    scale_code: Optional[Tensor] = None,
) -> Tensor:
    """
    Simplified MXINT8 quantization. Works on 2D tensors.
    If quantizing transposed 2D tensors, quantize along -2 axis.
    If quantizing non-transposed 2D tensors, quantize along -1 axis.

    Args:
      X:             Input tensor to quantize.
      code:          1D codebook tensor (e.g., from get_quantization_map).
      quant_max:     The maximum integer value (e.g., 127 for 8-bit).
      block_size:    Hardware vector length for tiling.
      axes:          Dimensions to tile/quantize over (e.g. [-1]).
      shared_exp_method: "max" or "none" for exponent sharing.
      force_scale_power_of_two: If True, snap scale to nearest power-of-two.
      scale_code:    Optional codebook for quantizing the scale itself.

    Returns:
      input_quantized: A tensor of the same shape as X, quantized to INT8.
      scale: tensor with scale factors
    """
    # 1. Prepare axes list
    if axes is None:
        raise ValueError("axes must be provided")
    axes = [ax + X.ndim if ax < 0 else ax for ax in axes]

    # 2. Tile input into blocks
    X_blocks, block_axes, orig_shape, padded_shape = _reshape_to_blocks(
        X, axes, block_size
    )

    # 3. Compute scale factor
    shared_axes = [ax + 1 for ax in block_axes]
    from quantized_training.mx_utils import _shared_exponents
    exp = _shared_exponents(
        X_blocks, method=shared_exp_method, axes=shared_axes, ebits=0
    )
    exp = exp - math.floor(math.log2(quant_max))
    scale = (2.0 ** exp).to(X.dtype)
    scale = torch.where(scale > 0, scale, torch.tensor(1.0, dtype=X.dtype, device=X.device))

    # 4. Quantize + dequantize
    input_quantized = vmap(X_blocks / scale, code).to(torch.int8)
    Xq_blocks = vmap(X_blocks / scale, code) * scale

    # 5. Undo tiling
    Xq = _undo_reshape_to_blocks(Xq_blocks, padded_shape, orig_shape, block_axes)
    input_quantized = _undo_reshape_to_blocks(input_quantized, padded_shape, orig_shape, block_axes)
    # Only squeeze dimensions that are equal to 1
    print(scale.shape)
    for dim in range(scale.dim()):
        if scale.size(dim) == 1:
            scale = scale.squeeze(dim)
            break
    return input_quantized, scale

if __name__ == "__main__":
    print("==============================")
    print("simple_quantize_mx: Test cases")
    print("==============================")
    matrix_row = 32
    matrix_col = 64

    # Load Q and K from files
    Q = torch.arange(matrix_row * matrix_col, dtype=torch.float32).reshape(matrix_row, matrix_col) / 10000
    K = torch.arange(matrix_row * matrix_col, matrix_row * matrix_col * 2, dtype=torch.float32).reshape(matrix_row, matrix_col) / 10000
    K = torch.transpose(K, 0, 1)

    print("Q", Q)
    print("K", K)

    print("Q shape: ", Q.shape)
    print("K shape: ", K.shape)

    print("Q quantizing...")
    quantized_Q, scale_Q = simple_mx_quantize(Q, axes=[-1], block_size=32)
    print("K quantizing...")
    quantized_K, scale_K = simple_mx_quantize(K, axes=[-2], block_size=32)
    # quantized_K = torch.transpose(quantized_K, 0, 1)
    # scale_K = torch.transpose(scale_K, 0, 1)

    # Load gold model values from files
    import os
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load gold model values from files with proper path
    with open(os.path.join(current_dir, 'q_mxint8.txt'), 'r') as f:
        gold_q_mxint8 = f.read().strip().split('\n')
        gold_q_mxint8 = [[float(val) for val in line.split(',')] for line in gold_q_mxint8]
        gold_q_mxint8 = torch.tensor(gold_q_mxint8, dtype=torch.int8)

    with open(os.path.join(current_dir, 'q_scale.txt'), 'r') as f:
        gold_q_scale = f.read().strip().split('\n')
        gold_q_scale = [[float(val) for val in line.split(',')] for line in gold_q_scale]
        gold_q_scale = torch.tensor(gold_q_scale, dtype=torch.float)

    with open(os.path.join(current_dir, 'k_mxint8.txt'), 'r') as f:
        gold_k_mxint8 = f.read().strip().split('\n')
        gold_k_mxint8 = [[float(val) for val in line.split(',')] for line in gold_k_mxint8]
        gold_k_mxint8 = torch.tensor(gold_k_mxint8, dtype=torch.int8)

    with open(os.path.join(current_dir, 'k_scale.txt'), 'r') as f:
        gold_k_scale = f.read().strip().split('\n')
        gold_k_scale = [[float(val) for val in line.split(',')] for line in gold_k_scale]
        gold_k_scale = torch.tensor(gold_k_scale, dtype=torch.float)
   
    print("Gold shapes:")
    print(gold_q_mxint8.shape)
    print(gold_q_scale.shape)
    print(gold_k_mxint8.shape)
    print(gold_k_scale.shape)

    print("Output shapes:")
    print(quantized_Q.shape)
    print(scale_Q.shape)
    print(quantized_K.shape)
    print(scale_K.shape)

    # Compare with our quantized values
    q_mxint8_match = torch.allclose(quantized_Q, gold_q_mxint8)
    q_scale_match = torch.allclose(scale_Q, gold_q_scale)
    k_mxint8_match = torch.allclose(quantized_K, gold_k_mxint8)
    k_scale_match = torch.allclose(scale_K, gold_k_scale)

    print(f"Q mxint8 matches gold model: {q_mxint8_match}")
    print(f"Q scale matches gold model: {q_scale_match}")
    print(f"K mxint8 matches gold model: {k_mxint8_match}")
    print(f"K scale matches gold model: {k_scale_match}")

    # If there are mismatches, show details
    if not q_mxint8_match:
        print("\nQ mxint8 differences:")
        diff_mask = quantized_Q.float() != gold_q_mxint8
        print(f"Number of different values: {diff_mask.sum().item()}")
        if diff_mask.sum() > 0:
            print("First few differences:")
            for i in range(min(5, diff_mask.sum().item())):
                indices = (diff_mask == True).nonzero()[i]
                print(f"Position {indices.tolist()}: Our value={quantized_Q[indices[0], indices[1]].item()}, Gold value={gold_q_mxint8[indices[0], indices[1]].item()}")

    if not q_scale_match:
        print("\nQ scale differences:")
        diff_mask = scale_Q != gold_q_scale
        print(f"Number of different values: {diff_mask.sum().item()}")
        if diff_mask.sum() > 0:
            print("First few differences:")
            for i in range(min(5, diff_mask.sum().item())):
                indices = (diff_mask == True).nonzero()[i]
                print(f"Position {indices.tolist()}: Our value={scale_Q[indices[0], indices[1]].item()}, Gold value={gold_q_scale[indices[0], indices[1]].item()}")

    if not k_mxint8_match:
        print("\nK mxint8 differences:")
        diff_mask = quantized_K.float() != gold_k_mxint8
        print(f"Number of different values: {diff_mask.sum().item()}")
        if diff_mask.sum() > 0:
            print("First few differences:")
            for i in range(min(5, diff_mask.sum().item())):
                indices = (diff_mask == True).nonzero()[i]
                print(f"Position {indices.tolist()}: Our value={quantized_K[indices[0], indices[1]].item()}, Gold value={gold_k_mxint8[indices[0], indices[1]].item()}")

    if not k_scale_match:
        print("\nK scale differences:")
        diff_mask = scale_K != gold_k_scale
        print(f"Number of different values: {diff_mask.sum().item()}")
        if diff_mask.sum() > 0:
            print("First few differences:")
            for i in range(min(5, diff_mask.sum().item())):
                indices = (diff_mask == True).nonzero()[i]
                print(f"Position {indices.tolist()}: Our value={scale_K[indices[0], indices[1]].item()}, Gold value={gold_k_scale[indices[0], indices[1]].item()}")