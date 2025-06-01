import os
import shutil
import torch
from quantized_training.codegen.mapping_utils import save_tensor
from typing import Union
import re
import math
from flash_attention.simple_quantize_mx import simple_mx_quantize

def write_flash_attention_layer_qkt(out_txt: str):
    # Writes flash_attention_step_0, 1, 2, ... to out_txt
    with open(out_txt, "w") as fh:
        fh.write("qkt_flash_attention" + "\n")
                        
def write_flash_attention_model_qkt(
    out_txt: str,
    dtype: str = "MXINT8",
    num_steps: int = 5,
    context_length: int = 64,
    block_size: int = 32, # for MXINT scale
    debug: bool = False,
) -> None:
    print("Writing model.txt")
    if dtype == "CFLOAT":
        dt_mat = dt_vec = "float32"; sz_mat = sz_vec = 4; need_mx = False
    elif dtype == "BFLOAT16":
        dt_mat = dt_vec = "bfloat16"; sz_mat = sz_vec = 2; need_mx = False
    elif dtype == "MXINT8":
        dt_mat, sz_mat = "int8", 1          # INT8 tiles
        dt_vec, sz_vec = "bfloat16", 2      # BF16 scratch
        need_mx = True
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    
    if debug:       
        print(f"need_mx: {need_mx}")
        print(f"dtype: {dtype}")

    cur_addr = 0
    lines: list[str] = []                   # accumulated proto text
    meta: dict[str, tuple] = {}             # name → (shape, dtype, addr)

    def alloc(shape, elem_size):
        nonlocal cur_addr
        addr = cur_addr
        cur_addr += math.prod(shape) * elem_size
        return addr

    def emit_tensor(kind, name, shape, dtype_name, addr, ind=""):
        pad = ind + "  "
        lines.append(f"{ind}{kind} {{")
        lines.append(f"{pad}node: \"{name}\"")
        for s in shape:
            lines.append(f"{pad}shape: {s}")
        lines.append(f"{pad}dtype: \"{dtype_name}\"")
        lines.append(f"{pad}memory {{")
        if addr:
            lines.append(f"{pad}  address: {addr}")
        lines.append(f"{pad}}}")
        lines.append(f"{ind}}}")

    def emit_input(*a, **kw):
        emit_tensor("inputs", *a, **kw)

    def emit_output(*a, **kw):
        emit_tensor("tensors", *a, ind="    ", **kw)   # 4‑sp indent inside outputs

    def kwargs_block(arg_key, node):
        shape, dt, addr = meta[node]
        if node.endswith("_step_0") and node.split("_step_")[0] in ("q_tile", "k_tile"):
            if len(shape) == 3 and shape[0] == 1:
                shape_to_emit = shape[1:]
            else:
                shape_to_emit = shape
        else:
            shape_to_emit = shape
        pad = "    "
        sub  = pad + "  "
        sub2 = sub + "  "
        txt = [f"{pad}kwargs {{",
               f'{sub}key: "{arg_key}"',
               f"{sub}value {{",
               f"{sub2}tensor {{",
               f'{sub2}  node: "{node}"']
        for s in shape_to_emit:
            txt.append(f"{sub2}  shape: {s}")
        txt.append(f'{sub2}  dtype: "{dt}"')
        txt.append(f"{sub2}  memory {{")
        if addr:
            txt.append(f"{sub2}    address: {addr}")
        txt.append(f"{sub2}  }}")   # memory
        txt.append(f"{sub2}}}")     # tensor
        txt.append(f"{sub}}}")     # value
        txt.append(f"{pad}}}")      # kwargs
        return txt

    def kwargs_scalar_block(key, val, indent="    "):
        pad = indent
        sub = indent + "  "
        if isinstance(val, int):
            return [
                f"{pad}kwargs {{",
                f'{sub}key: "{key}"',
                f"{sub}value {{",
                f"{sub}  int_value: {val}",
                f"{sub}}}",
                f"{pad}}}"
            ]
        elif isinstance(val, float):
            return [
                f"{pad}kwargs {{",
                f'{sub}key: "{key}"',
                f"{sub}value {{",
                f"{sub}  float_value: {val}",
                f"{sub}}}",
                f"{pad}}}"
            ]
        else:
            raise ValueError(f"Unsupported scalar kwarg type for key: {key}")


    # map kw‑key → logical tensor name
    if need_mx:
        QKT_KWARG_MAP = {
            "q_tensor": "q_tensor",
            "q_tensor_scale": "q_tensor_scale",
            "k_tensor": "k_tensor",
            "k_tensor_scale": "k_tensor_scale",
        }
        PV_KWARG_MAP = {
            "p_tensor": "p_tensor",
            "p_tensor_scale": "p_tensor_scale",
            "v_tensor": "v_tensor",
            "v_tensor_scale": "v_tensor_scale",
            "sum": "sum",
            "rescale": "rescale",
        }
    else:
        QKT_KWARG_MAP = {
            "q_tensor": "q_tensor",
            "k_tensor": "k_tensor",
        }
        PV_KWARG_MAP = {
            "p_tensor": "p_tensor",
            "v_tensor": "v_tensor",
            "sum": "sum",
            "rescale": "rescale",
        }

    # TODO: fix the hardcoding of these
    QKT_SCALAR_KWARG_MAP = {
        "key_dim": 0.125,
        "context_length": context_length,
        "num_heads": 12,
        "quantize_mx": 1,
        "block_size": block_size,
        "quant_max": 127.0,
        "force_scale_power_of_two": 1,
        "mask_mode": 1,
    }
    PV_SCALAR_KWARG_MAP = {
        "context_length": context_length,
        "num_heads": 12,
        "quantize_mx": 0,
    }

    
    def add_matrix(name, shape, axis=-1):
        addr = alloc(shape, sz_mat)
        emit_input(name, shape, dt_mat, addr)
        meta[name] = (shape, dt_mat, addr)
        if need_mx:
            if axis == -2:
                # Divide block_size on the second-to-last axis
                sshape = shape[:-2] + (shape[-2] // block_size,) + shape[-1:]
            else:  # axis == -1 or default
                # Divide block_size on the last axis
                sshape = shape[:-1] + (shape[-1] // block_size,)
            saddr = alloc(sshape, 1)
            emit_input(f"{name}_scale", sshape, "fp8_e8m0", saddr)
            meta[f"{name}_scale"] = (sshape, "fp8_e8m0", saddr)

    def add_out(name, shape, dtype, size):
        addr = alloc(shape, size)
        emit_output(name, shape, dtype, addr)
        meta[name] = (shape, dtype, addr)

    # ----- 1.  INPUTS ----------------------------------------------------
    # Add Q, K, V inputs
    add_matrix("q_tensor", (context_length, 64))
    add_matrix("k_tensor", (context_length, 64))
    add_matrix("v_tensor", (context_length, 64))

    # ----- a.  QKT Arguments  --------------------------------------------
    lines.append("ops {")
    lines.append("  op {")
    lines.append('    name: "qkt_flash_attention"')
    lines.append('    op: "call_function"')
    lines.append('    target: "qkt_flash_attention"')

    # Generate kwargs for input tensors
    for k, base in QKT_KWARG_MAP.items():
        if base in meta:
            lines += kwargs_block(k, base)

    for k, base in QKT_SCALAR_KWARG_MAP.items():
        lines += kwargs_scalar_block(k, base)

    lines.append("  }")        # close op
    lines.append("  outputs {")

    # Add outputs
    add_out("qkt_tensor", (context_length, context_length), dt_vec, sz_vec)
    add_out("p_tensor", (context_length, context_length), dt_mat, sz_mat)
    
    if need_mx:
        saddr_shape = (context_length, context_length // block_size)
        addr = alloc(saddr_shape, 1)
        emit_output("p_tensor_scale", saddr_shape, "fp8_e8m0", addr)
        meta["p_tensor_scale"] = (saddr_shape, "fp8_e8m0", addr)
    
    add_out("max", ((num_steps - 1) * block_size,), dt_vec, sz_vec)
    add_out("sum", ((num_steps - 1) * block_size,), dt_vec, sz_vec)
    add_out("rescale", ((num_steps - 1) * block_size,), dt_vec, sz_vec)

    lines.append("  }")        # close outputs
    lines.append("}")          # close ops
    lines.append("")           # blank line between steps

    # ----- b.  PV Arguments  ---------------------------------------------
    lines.append("ops {")
    lines.append("  op {")
    lines.append('    name: "pv_flash_attention"')
    lines.append('    op: "call_function"')
    lines.append('    target: "pv_flash_attention"')

    # Generate kwargs for input tensors
    for k, base in PV_KWARG_MAP.items():
        if base in meta:
            lines += kwargs_block(k, base)

    for k, base in PV_SCALAR_KWARG_MAP.items():
        lines += kwargs_scalar_block(k, base)

    lines.append("  }")        # close op
    lines.append("  outputs {")

    # Add outputs
    add_out("self_attention_tensor", (context_length, 64), dt_vec, sz_vec)
    
    # TODO: add support for quantizing self_attention
    # if need_mx:
    #     saddr_shape = (context_length, context_length // block_size)
    #     addr = alloc(saddr_shape, 1)
    #     emit_output("p_tensor_scale", saddr_shape, "fp8_e8m0", addr)
    #     meta["p_tensor_scale"] = (saddr_shape, "fp8_e8m0", addr)

    lines.append("  }")        # close outputs
    lines.append("}")          # close ops
    lines.append("")           # blank line between steps

    # ----- 4.  write file ----------------------------------------------------
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf‑8") as fh:
        fh.write("\n".join(lines))

# ────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────── Utils ─────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
       
def write_flash_attention_tilings_qkt(out_txt: str):
  with open (out_txt, "w") as fh:
    fh.write("")

def bin_name(tensor_name: str, dir: str) -> str:
    return os.path.join(dir, f"{tensor_name}.bin")

def save(t: torch.Tensor, name: str, dir: str, dtype: str, tile_size: int):
    if dtype == "CFLOAT" or dtype == "BFLOAT16":
        save_tensor(t, bin_name(name, dir))
    elif dtype == "MXINT8": 
        # QK^T and PV are quantized to MXINT8
        quantized_tensors = ["q_tensor", "k_tensor", "v_tensor", "p_tensor"]   
        if name in quantized_tensors:
            if name == "k_tensor":
                q, scale = simple_mx_quantize(t, axes=[-2], block_size=tile_size)
            else:
                q, scale = simple_mx_quantize(t, axes=[-1], block_size=tile_size)
            save_tensor(q, bin_name(name, dir)) 
            save_tensor(scale, bin_name(name + "_scale", dir)) 
        else:
            # Other tensors are saved as bfloat16
            save_tensor(t.to(torch.bfloat16), bin_name(name, dir))

# ────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────── Flash Attention ────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────

def flash_attention_tile_step_qkt(
    global_step: int,
    total_steps: int,
    Q_tile: Union[torch.Tensor, None],
    K_tile: Union[torch.Tensor, None],
    V_tile: Union[torch.Tensor, None],
    dk: float,
    p_input: torch.Tensor,
    max_input: torch.Tensor,
    sum_input: torch.Tensor,
    rescale_input: torch.Tensor,
    o_input: torch.Tensor,
    output_dir_root: str,
    txt: bool = False,
    debug: bool = False,
    dtype: str = "CFLOAT"
):
    """
    Flash-attention aggregator for one tile step, using a *global* step index
    to decide skip logic:
      - If global_step == 0      => skip partial p@V
      - If global_step == total_steps-1 => skip QK^T+softmax
      - Otherwise, do both.

    We store .bin files in output_dir_root/qkt_flash_attention/tensor_files.
    """

    # Are we on the very first overall step?
    on_first_step = (global_step == 0)
    # Are we on the very last overall step?
    on_last_step = (global_step == (total_steps - 1))

    print("=====================================================================================")
    print(f"FLASH ATTENTION QKT OPERATION (global_step={global_step}, total_steps={total_steps})")
    print(f" first_step={on_first_step}, last_step={on_last_step}")
    print("=====================================================================================\n")

    # Print aggregator inputs for debugging
    if debug:
      # Print shapes and values with None checks for all tensors
      if Q_tile is not None:
          print(f"Q_tile shape: {Q_tile.shape}, values:\n{Q_tile}")
      else:
          print("Q_tile is None")
          
      if K_tile is not None:
          print(f"K_tile shape: {K_tile.shape}, values:\n{K_tile}")
      else:
          print("K_tile is None")
          
      if V_tile is not None:
          print(f"V_tile shape: {V_tile.shape}, values:\n{V_tile}")
      else:
          print("V_tile is None")
          
      if p_input is not None:
          print(f"p_input shape: {p_input.shape}, values:\n{p_input}")
      else:
          print("p_input is None")
          
      if max_input is not None:
          print(f"max_input shape: {max_input.shape}, values:\n{max_input}")
      else:
          print("max_input is None")
          
      if sum_input is not None:
          print(f"sum_input shape: {sum_input.shape}, values:\n{sum_input}")
      else:
          print("sum_input is None")
          
      if rescale_input is not None:
          print(f"rescale_input shape: {rescale_input.shape}, values:\n{rescale_input}")
      else:
          print("rescale_input is None")
          
      if o_input is not None:
          print(f"o_input shape: {o_input.shape}, values:\n{o_input}\n")
      else:
          print("o_input is None\n")
          
      # Print just the shapes with None checks
      print(f"Q_tile shape: {Q_tile.shape if Q_tile is not None else None}")
      print(f"K_tile shape: {K_tile.shape if K_tile is not None else None}")
      print(f"V_tile shape: {V_tile.shape if V_tile is not None else None}")
      print(f"p_input shape: {p_input.shape if p_input is not None else None}")
      print(f"max_input shape: {max_input.shape if max_input is not None else None}")
      print(f"sum_input shape: {sum_input.shape if sum_input is not None else None}")
      print(f"rescale_input shape: {rescale_input.shape if rescale_input is not None else None}")
      print(f"o_input shape: {o_input.shape if o_input is not None else None}\n")

    # Prepare placeholders
    qkt_tile = None
    max_output = None
    rescale_output = None
    p_tile = None
    sum_output = None
    o_output = None
    pv_output = None
    
    step_dir = os.path.join(output_dir_root, "tensor_files")

    os.makedirs(step_dir, exist_ok=True)

    print("----- Operations & Outputs This Step -----")

    # ------------------------------------------
    # 1) QK^T + softmax => skip ONLY if last step
    # ------------------------------------------
    if not on_last_step:
        print("Performing QK^T + softmax...")

        assert Q_tile is not None
        assert K_tile is not None

        qkt_tile = Q_tile @ K_tile.T / dk

        row_max = torch.max(qkt_tile, dim=-1, keepdim=True).values
        max_output = torch.maximum(max_input, row_max)
        rescale_output = torch.exp(max_input - max_output)
        
        normalized_qkt = qkt_tile - max_output
        p_tile = torch.exp(normalized_qkt)

        row_sum = torch.sum(p_tile, dim=-1, keepdim=True)
        sum_output = rescale_output * sum_input + row_sum
        
        if debug:
          print(f"QK^T shape: {qkt_tile.shape}, values:\n{qkt_tile}")
          print(f"Row max shape: {row_max.shape}, values:\n{row_max}")
          print(f"max_output shape: {max_output.shape}, values:\n{max_output}")
          print(f"Rescale out shape: {rescale_output.shape}, values:\n{rescale_output}")
          print(f"Normalized QK^T shape: {normalized_qkt.shape}, values:\n{normalized_qkt}")
          print(f"p_tile shape: {p_tile.shape}, values:\n{p_tile}")
          print(f"Row sum shape: {row_sum.shape}, values:\n{row_sum}")
          print(f"sum_output shape: {sum_output.shape}, values:\n{sum_output}")

    else:
        print("Skipping QK^T + softmax on the very last step => no bin files for that.")

    # ----------------------------------------------
    # 2) Partial p@V => skip ONLY if first step
    # ----------------------------------------------
    if not on_first_step:
        print("Performing partial p@V + update o_output...")

        assert V_tile is not None

        pv_output = p_input @ V_tile

        assert o_input is not None
        assert rescale_input is not None

        o_rescaled = o_input * rescale_input
        o_output = o_rescaled + pv_output

        if debug:
          print(f"p@V shape: {pv_output.shape}, values:\n{pv_output}")
          print(f"o_rescaled shape: {o_rescaled.shape}, values:\n{o_rescaled}")
      
    else:
        print("Skipping partial p@V on the very first step => no bin files for that.")

    if debug:
      print("\n===== END OF STEP OUTPUTS =====\n")
      print(f"qkt_tile: {qkt_tile}")
      print(f"max_output:\n{max_output}")
      print(f"rescale_output:\n{rescale_output}")
      print(f"p_tile:\n{p_tile}")
      print(f"sum_output:\n{sum_output}")
      print(f"pv_output:\n{pv_output}")
      print(f"o_output:\n{o_output}\n\n")
      print(f"qkt_tile: {qkt_tile is not None}")
      print(f"max_output: {max_output is not None}")
      print(f"rescale_output: {rescale_output is not None}")
      print(f"p_tile: {p_tile is not None}")
      print(f"sum_output: {sum_output is not None}")
      print(f"pv_output: {pv_output is not None}")
      print(f"o_output: {o_output is not None}")
    
    return qkt_tile, p_tile, max_output, sum_output, rescale_output, pv_output, o_output

def run_flash_attention_tiled_qkt(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    tile_size: int,
    output_dir_root: str,
    dtype: str = "MXINT8",
    debug: bool = False,
):
    """
    - We split Q, K, V by rows into tile_size blocks.
    - total_steps = (#Q_tiles * #K_tiles).
    - global_step runs from 0..(total_steps - 1):
       * If global_step=0 => skip partial p@V
       * If global_step=total_steps-1 => skip QK^T+softmax
       * Otherwise => do both
    - Each Q tile has its own aggregator states, re-initialized at the start
      of that tile's loop.

    Implementation:
      for i in range(#Q_tiles):
        reset aggregator states
        for j in range(#K_tiles):
          call flash_attention_tile_step(...) with global_step
          global_step += 1
    """

    os.makedirs(output_dir_root, exist_ok=True)

    # Split row-wise
    Q_tiles = torch.split(Q, tile_size, dim=0)
    K_tiles = torch.split(K, tile_size, dim=0)
    V_tiles = torch.split(V, tile_size, dim=0)

    Q_tile_queue = []
    K_tile_queue = []
    V_tile_queue = []

    num_Q_tiles = len(Q_tiles)
    num_K_tiles = len(K_tiles)
    total_steps = num_Q_tiles * num_K_tiles + 1
    dk = math.sqrt(Q.size(1))

    global_step = 0
    results_o = []

    # Creates pipeline of Q, K, V tiles
    for i, Q_tile in enumerate(Q_tiles):
        for j, (K_tile, V_tile) in enumerate(zip(K_tiles, V_tiles)):
            Q_tile_queue.append(Q_tile)
            K_tile_queue.append(K_tile)
            V_tile_queue.append(V_tile)
    Q_tile_queue.append(None)
    K_tile_queue.append(None)
    V_tile_queue.insert(0, None)

    sum_queue = []

    qkt_return = []
    p_return = []
    max_return = []
    sum_return = []
    rescale_return = []

    # Initialize statistic values
    def initialize_stats_worker_1(tile_rows, col):
        max_input = torch.full((tile_rows, 1), float('-inf'))
        sum_input = torch.zeros((tile_rows, 1))
        return max_input, sum_input
    
    def initialize_stats_worker_2(tile_rows, col):
        p_input = torch.zeros((tile_rows, tile_rows))
        rescale_input = torch.zeros((tile_rows, 1))
        o_input = torch.zeros((tile_rows, col))
        return p_input, rescale_input, o_input

    for i in range(total_steps):
        if i == 0:
            # init both workers
            max_input, sum_input = initialize_stats_worker_1(Q_tile.size(0), Q_tile.size(1))
            p_input, rescale_input, o_input = initialize_stats_worker_2(Q_tile.size(0), Q_tile.size(1))
        elif i == total_steps - 1 and total_steps == 2:
            # LAST STEP: make sure o_input is fresh
            print(f"Step {i}: reinitializing worker 2 for final step")
            _, rescale_input, o_input = initialize_stats_worker_2(tile_size, V.size(1))
        elif i % num_Q_tiles == 0:
            print(f"Step {i}: reinitializing worker 1 statistics")
            max_input, sum_input = initialize_stats_worker_1(Q_tile.size(0), Q_tile.size(1))
        elif i % num_K_tiles == 1:
            print(f"Step {i}: reinitializing worker 2 statistics")
            _, _, o_input = initialize_stats_worker_2(Q_tile.size(0), Q_tile.size(1))

        (
            qkt_tile,
            p_input,
            max_input,
            sum_input,
            rescale_input,
            pv_output,
            o_input
        ) = flash_attention_tile_step_qkt(
            global_step=global_step,
            total_steps=total_steps,
            Q_tile=Q_tile_queue[i],
            K_tile=K_tile_queue[i],
            V_tile=V_tile_queue[i],
            dk=dk,
            p_input=p_input,
            max_input=max_input,
            sum_input=sum_input,
            rescale_input=rescale_input,
            o_input=o_input,
            output_dir_root=output_dir_root,
            dtype=dtype
        )

        sum_queue.append(sum_input)

        if qkt_tile is not None:
            qkt_return.append(qkt_tile)
        if p_input is not None:
            p_return.append(p_input)
        if max_input is not None:
            max_return.append(max_input)
        if sum_input is not None:
            sum_return.append(sum_input)
        if rescale_input is not None:
            rescale_return.append(rescale_input)

        if i % num_Q_tiles == 0 and i != 0:
            results_o.append(o_input / sum_queue[i-1])
        
        global_step += 1

    if debug:
        print(f"qkt_return size: {len(qkt_return)}")
        print(f"p_return size: {len(p_return)}")
        print(f"max_return size: {len(max_return)}")
        print(f"sum_return size: {len(sum_return)}")
        print(f"rescale_return size: {len(rescale_return)}")

    # STITCHING TOGETHER QKT TILES
    # Stitch together the QKT tiles into a complete QKT matrix
    # For a context length of n, we'll have n/tile_size rows and columns of tiles
    # Check if all tensors in qkt_return are not None and print the values
    iter_values = [t is not None for t in qkt_return]
    iter = all(iter_values)
    print(f"Values of each tensor check: {iter_values}")
    print(f"all(t is not None for t in qkt_return): {iter}")
    if qkt_return and all(t is not None for t in qkt_return):
        # Calculate dimensions
        tile_dim = qkt_return[0].size(0) 
        num_tiles_per_row = int(math.sqrt(len(qkt_return)))
        context_length = num_tiles_per_row * tile_dim
        
        # Initialize the full QKT matrix
        full_qkt = torch.zeros((context_length, context_length), device=qkt_return[0].device)
        
        # Place each tile in its correct position
        for idx, tile in enumerate(qkt_return):
            if tile is None:
                continue
                
            # Calculate row and column position for this tile
            row_idx = (idx // num_tiles_per_row) * tile_dim
            col_idx = (idx % num_tiles_per_row) * tile_dim
            
            # Place the tile in the full matrix
            full_qkt[row_idx:row_idx+tile_dim, col_idx:col_idx+tile_dim] = tile
            
        qkt_return = full_qkt
        if debug:
            print(f"qkt_return shape: {qkt_return.shape}")
        
    # STITCHING TOGETHER P TILES
    # Stitch together the P tiles into a complete P matrix
    # For a context length of n, we'll have n/tile_size rows and columns of tiles
    if p_return and all(t is not None for t in p_return):
        # Calculate dimensions
        tile_dim = p_return[0].size(0)
        num_tiles_per_row = int(math.sqrt(len(p_return)))
        context_length = num_tiles_per_row * tile_dim
        
        # Initialize the full P matrix
        full_p = torch.zeros((context_length, context_length), device=p_return[0].device)
        
        # Place each tile in its correct position
        for idx, tile in enumerate(p_return):
            if tile is None:
                continue
                
            # Calculate row and column position for this tile
            row_idx = (idx // num_tiles_per_row) * tile_dim
            col_idx = (idx % num_tiles_per_row) * tile_dim
            
            # Place the tile in the full matrix
            full_p[row_idx:row_idx+tile_dim, col_idx:col_idx+tile_dim] = tile
            
        p_return = full_p
        if debug:
            print(f"p_return shape: {p_return.shape}")

    # STITCHING TOGETHER MAX, SUM, AND RESCALE TENSORS
    # These are row vectors for each tile, so we just need to concatenate them
    if max_return and all(t is not None for t in max_return):
        # Concatenate the max vectors
        full_max = torch.cat(max_return, dim=0)
        max_return = full_max
        if debug:
            print(f"max_return shape: {max_return.shape}")
    
    if sum_return and all(t is not None for t in sum_return):
        # Concatenate the sum vectors
        full_sum = torch.cat(sum_return, dim=0)
        sum_return = full_sum
        if debug:
            print(f"sum_return shape: {sum_return.shape}")
    
    if rescale_return and all(t is not None for t in rescale_return):
        # Concatenate the rescale vectors
        full_rescale = torch.cat(rescale_return, dim=0)
        rescale_return = full_rescale
        if debug:
            print(f"rescale_return shape: {rescale_return.shape}")

    # Combine all partial O blocks
    final_O = torch.cat(results_o, dim=0)

    # Save all the tensors to the tensor_files directory
    dir = os.path.join(output_dir_root, "tensor_files")
    os.makedirs(dir, exist_ok=True)

    # Save all the input tensors
    save(Q,               "q_tensor", dir, dtype, tile_size)
    save(K,               "k_tensor", dir, dtype, tile_size)
    save(V,               "v_tensor", dir, dtype, tile_size)
    # Save all the results tensors
    save(qkt_return,      "qkt_tensor", dir, dtype, tile_size)
    save(p_return,        "p_tensor", dir, dtype, tile_size)
    save(final_O,         "self_attention_tensor", dir, dtype, tile_size)
    save(max_return,      "max", dir, dtype, tile_size)
    save(sum_return,      "sum", dir, dtype, tile_size)
    save(rescale_return,  "rescale", dir, dtype, tile_size)
    

    # Save all the tensors to the tensor_files directory
    if debug:
        with open(os.path.join(dir, "q_tensor.txt"), 'w') as f:
            f.write(str(Q))
        with open(os.path.join(dir, "k_tensor.txt"), 'w') as f:
            f.write(str(K))
        with open(os.path.join(dir, "v_tensor.txt"), 'w') as f:
            f.write(str(V))
        with open(os.path.join(dir, "qkt_tensor.txt"), 'w') as f:
            f.write(str(qkt_return))
        with open(os.path.join(dir, "p_tensor.txt"), 'w') as f:
            f.write(str(p_return))
        with open(os.path.join(dir, "self_attention_tensor.txt"), 'w') as f:
            f.write(str(final_O))
            
        with open(os.path.join(dir, "max.txt"), 'w') as f:
            f.write(str(max_return))
        with open(os.path.join(dir, "sum.txt"), 'w') as f:
            f.write(str(sum_return))
        with open(os.path.join(dir, "rescale.txt"), 'w') as f:
            f.write(str(rescale_return))

    # # Write model.txt, layers.txt, and tilings.txtpb
    model_txt = os.path.join(output_dir_root, "model.txt")
    write_flash_attention_model_qkt(model_txt, dtype, num_steps=total_steps, context_length=context_length, block_size=tile_size, debug=debug)

    layers_txt = os.path.join(output_dir_root, "layers.txt")
    write_flash_attention_layer_qkt(layers_txt)
    
    # Add environment variables for testing
    ic_dim = os.environ["IC_DIMENSION"]
    oc_dim = os.environ["OC_DIMENSION"]
    input_buffer_size = 1024
    weight_buffer_size = 1024
    accum_buffer_size = 1024
    double_buffered_accum_buffer = os.environ["DOUBLE_BUFFERED_ACCUM_BUFFER"]

    if debug:
        print(f"ic_dim: {ic_dim}")
        print(f"oc_dim: {oc_dim}")
        print(f"input_buffer_size: {input_buffer_size}")
        print(f"weight_buffer_size: {weight_buffer_size}")
        print(f"accum_buffer_size: {accum_buffer_size}")
        print(f"double_buffered_accum_buffer: {double_buffered_accum_buffer}")

    tilings_txt = os.path.join(output_dir_root, f"{ic_dim}x{oc_dim}_{input_buffer_size}x{weight_buffer_size}x{accum_buffer_size}_{double_buffered_accum_buffer}", "tilings.txtpb")
    os.makedirs(os.path.dirname(tilings_txt), exist_ok=True)
    write_flash_attention_tilings_qkt(tilings_txt)
          
    return final_O

def vanilla_attention(Q, K, V):
    dk = math.sqrt(Q.size(1))
    return ((Q @ K.T) / dk).softmax(dim=-1) @ V
    # return ((Q @ K.T)).softmax(dim=-1) @ V

def compare_results(final_O, final_O_vanilla):
    # Compare the results between tiled flash attention and vanilla attention
    print("\n--- Comparing tiled flash attention vs vanilla attention ---")
    
    # Calculate absolute difference
    abs_diff = torch.abs(final_O - final_O_vanilla)
    
    # Calculate statistics
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    mse = torch.mean((final_O - final_O_vanilla) ** 2).item()
    
    # Calculate relative error where values are not too small
    mask = torch.abs(final_O_vanilla) > 1e-6
    if mask.any():
        rel_error = torch.abs((final_O - final_O_vanilla)[mask] / final_O_vanilla[mask])
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
    else:
        max_rel_error = float('nan')
        mean_rel_error = float('nan')
    
    # Print comparison results
    print(f"Maximum absolute difference: {max_diff:.8e}")
    print(f"Mean absolute difference:    {mean_diff:.8e}")
    print(f"Mean squared error:          {mse:.8e}")
    print(f"Maximum relative error:      {max_rel_error:.8e}")
    print(f"Mean relative error:         {mean_rel_error:.8e}")
    
    # Check if results are close enough
    rtol = 1e-5  # relative tolerance
    atol = 1e-7  # absolute tolerance
    is_close = torch.allclose(final_O, final_O_vanilla, rtol=rtol, atol=atol)
    print(f"\nResults are {'close' if is_close else 'NOT close'} within tolerance (rtol={rtol}, atol={atol})")

def generate_flash_attention_data_qkt(context_length=64, 
                                      output_dir_root="test/compiler/networks/flash_attention/CFLOAT/", 
                                      random=True, 
                                      dtype="CFLOAT",
                                      tile_size=32):
    print("Generating flash attention gold data.")
    print("Context length:", context_length)

    torch.set_default_dtype(torch.float32)
    
    matrix_row = context_length
    matrix_col = 64
    random = random

    if not random:
        Q = torch.arange(matrix_row * matrix_col).reshape(matrix_row, matrix_col) / 10000
        K = torch.arange(matrix_row * matrix_col, matrix_row * matrix_col * 2).reshape(matrix_row, matrix_col) / 10000
        V = torch.arange(matrix_row * matrix_col).reshape(matrix_row, matrix_col) / 10000
    else:
        Q = torch.randn(matrix_row, matrix_col)
        K = torch.randn(matrix_row, matrix_col)
        V = torch.randn(matrix_row, matrix_col)

    # Delete all previous flash attention tensor_files
    # tensor_files_dir = os.path.join(output_dir_root, "tensor_files")
    # if os.path.exists(tensor_files_dir):
    #     shutil.rmtree(tensor_files_dir)
    
    final_O = run_flash_attention_tiled_qkt(Q, K, V, tile_size, output_dir_root, dtype)
    print("Tiled Attention:", final_O.shape)
    print(final_O)

    final_O_vanilla = vanilla_attention(Q, K, V)
    print("Vanilla Attention:", final_O_vanilla.shape)
    print(final_O_vanilla)

    compare_results(final_O, final_O_vanilla)
    return final_O_vanilla, final_O
