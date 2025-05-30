import os
import shutil
import torch
from quantized_training.codegen.mapping_utils import save_tensor
from typing import Union
import re
import math
from flash_attention.simple_quantize_mx import simple_mx_quantize

def write_flash_attention_layers(out_txt: str, num_steps: int = 5):
    
    # Writes flash_attention_step_0, 1, 2, ... to out_txt
    with open(out_txt, "w") as fh:
        for s in range(num_steps):
            fh.write("flash_attention_step_" + str(s) + "\n")
                        
def write_flash_attention_model(
    out_txt: str,
    num_steps: int = 5,
    dtype: str = "MXINT8",
    block_size: int = 32, # for MXINT scale
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
                shape_to_emit = shape[1:]          # (32, 64)
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
        txt.append(f"{sub} }}")     # value
        txt.append(f"{pad}}}")      # kwargs
        return txt

    # map kw‑key → logical tensor name (without “_step_n” suffix)
    if need_mx:
        KWARG_MAP = {
            "query_tile":   "q_tile",
            "query_tile_scale": "q_tile_scale",
            "key_tile":     "k_tile",
            "key_tile_scale": "k_tile_scale",
            "value_tile":   "v_tile",
            "value_tile_scale": "v_tile_scale",
            "softmax_input":"softmax_input",
            "softmax_input_scale": "softmax_input_scale",
            "max_input":    "max_input",
            "sum_input":    "sum_input",
            "rescale_input":"rescale_input",
            "o_input":      "o_input",
        }
    else:
        KWARG_MAP = {
            "query_tile":   "q_tile",
            "key_tile":     "k_tile",
            "value_tile":   "v_tile",
            "softmax_input":"softmax_input",
            "max_input":    "max_input",
            "sum_input":    "sum_input",
            "rescale_input":"rescale_input",
            "o_input":      "o_input",
        }
    

    # ───────────────────────── per‑step generation
    for step in range(num_steps):
        first = (step == 0)
        last  = (step == num_steps - 1)

        # ----- 1.  INPUTS ----------------------------------------------------
        def add_matrix(name, shape, axis=-1):
            addr = alloc(shape, sz_mat)
            emit_input(f"{name}_step_{step}", shape, dt_mat, addr)
            meta[f"{name}_step_{step}"] = (shape, dt_mat, addr)
            if need_mx:
                if axis == -2:
                    # Divide block_size on the second-to-last axis
                    sshape = shape[:-2] + (shape[-2] // block_size,) + shape[-1:]
                else:  # axis == -1 or default
                    # Divide block_size on the last axis
                    sshape = shape[:-1] + (shape[-1] // block_size,)
                saddr  = alloc(sshape, 1)
                emit_input(f"{name}_scale_step_{step}", sshape, "fp8_e8m0", saddr)
                meta[f"{name}_scale_step_{step}"] = (sshape, "fp8_e8m0", saddr)

        if not last:                       # Q, K appear on every step except final PV‑only
            add_matrix("q_tile", (1, 32, 64) if first else (32, 64))
            add_matrix("k_tile", (1, 32, 64) if first else (32, 64), axis=-1)

        if not first:                      # V & p‑in missing only on very first step
            add_matrix("v_tile",          (32, 64), axis=-2)
            add_matrix("softmax_input",   (32, 32))

        # vector‑unit scratch
        def add_vec(name, shape):
            addr = alloc(shape, sz_vec)
            emit_input(f"{name}_step_{step}", shape, dt_vec, addr)
            meta[f"{name}_step_{step}"] = (shape, dt_vec, addr)

        if not last:
            add_vec("max_input",     (1, 32))
            add_vec("sum_input",     (1, 32))
        if not first:
            add_vec("rescale_input", (1, 32))
            add_vec("o_input",       (32, 64))

        # ----- 2.  OP + KWARGS ----------------------------------------------
        lines.append("ops {")
        lines.append("  op {")
        lines.append(f'    name: "flash_attention_step_{step}"')
        lines.append('    op: "call_function"')
        lines.append('    target: "flash_attention"')

        # generate kwargs only for tensors that exist this step
        print(f"meta: {meta}")
        for k, base in KWARG_MAP.items():
            full = f"{base}_step_{step}"
            print(f"full: {full}")
            if full in meta:
                lines += kwargs_block(k, full)

        lines.append("  }")        # close op

        # ----- 3.  OUTPUTS ---------------------------------------------------
        lines.append("  outputs {")

        def add_out(name, shape, dtype, size):
            addr = alloc(shape, size)
            emit_output(name, shape, dtype, addr)
            meta[name] = (shape, dtype, addr)

        if not last:
            add_out(f"qkt_tile_step_{step}",        (32, 32), dt_vec, sz_vec)

        if not first:
            add_out(f"pv_output_step_{step}",       (32, 64), dt_vec, sz_vec)
            add_out(f"o_output_step_{step}",        (32, 64), dt_vec, sz_vec)

        if not last:
            add_out(f"softmax_output_step_{step}", (32, 32), dt_mat, sz_mat)
            if need_mx:
                saddr_shape = (32, 32 // block_size)
                addr = alloc(saddr_shape, 1)
                emit_output(f"softmax_output_scale_step_{step}",
                            saddr_shape, "fp8_e8m0", addr)
                meta[f"softmax_output_scale_step_{step}"] = (saddr_shape, "fp8_e8m0", addr)

            add_out(f"max_output_step_{step}",    (32,),    dt_vec, sz_vec)
            add_out(f"sum_output_step_{step}",    (32,),    dt_vec, sz_vec)
            add_out(f"rescale_output_step_{step}",(1, 32),  dt_vec, sz_vec)

        lines.append("  }")        # close outputs
        lines.append("}")          # close ops
        lines.append("")           # blank line between steps

    # ----- 4.  write file ----------------------------------------------------
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf‑8") as fh:
        fh.write("\n".join(lines))

       
def write_flash_attention_tilings(out_txt: str):
  with open (out_txt, "w") as fh:
    fh.write("")

def flash_attention_tile_step(
    global_step: int,
    total_steps: int,
    Q_tile: Union[torch.Tensor, None],
    K_tile: Union[torch.Tensor, None],
    V_tile: Union[torch.Tensor, None],
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

    We store .bin files in output_dir_root/flash_attention_{global_step}/tensor_files.
    """

    # Are we on the very first overall step?
    on_first_step = (global_step == 0)
    # Are we on the very last overall step?
    on_last_step = (global_step == (total_steps - 1))

    print("======================================================")
    print(f"FLASH ATTENTION (global_step={global_step}, total_steps={total_steps})")
    print(f" first_step={on_first_step}, last_step={on_last_step}")
    print("======================================================\n")

    # Print aggregator inputs for debugging
    if debug:
      print(f"Q_tile shape: {Q_tile.shape}, values:\n{Q_tile}")
      print(f"K_tile shape: {K_tile.shape}, values:\n{K_tile}")
      print(f"V_tile shape: {V_tile.shape}, values:\n{V_tile}")
      print(f"p_input shape: {p_input.shape}, values:\n{p_input}")
      print(f"max_input shape: {max_input.shape}, values:\n{max_input}")
      print(f"sum_input shape: {sum_input.shape}, values:\n{sum_input}")
      print(f"rescale_input shape: {rescale_input.shape}, values:\n{rescale_input}")
      print(f"o_input shape: {o_input.shape}, values:\n{o_input}\n")
      print(f"Q_tile shape: {Q_tile.shape if Q_tile is not None else None}")
      print(f"K_tile shape: {K_tile.shape if K_tile is not None else None}")
      print(f"V_tile shape: {V_tile.shape if V_tile is not None else None}")
      print(f"p_input shape: {p_input.shape}")
      print(f"max_input shape: {max_input.shape}")
      print(f"sum_input shape: {sum_input.shape}")
      print(f"rescale_input shape: {rescale_input.shape}")
      print(f"o_input shape: {o_input.shape}\n")

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
    
    def bin_name(tensor_name: str) -> str:
        return os.path.join(step_dir, f"{tensor_name}_step_{global_step}.bin")
    
    def save(t: torch.Tensor, name: str):
        if dtype == "CFLOAT" or dtype == "BFLOAT16":
            save_tensor(t, bin_name(name))
        elif dtype == "MXINT8": 
            # QK^T and PV are quantized to MXINT8
            quantized_tensors = ["q_tile", "k_tile", "v_tile", "softmax_input", "softmax_output"]   
            if name in quantized_tensors:
                if name == "k_tile" or name == "v_tile":
                    q, scale = simple_mx_quantize(t, axes=[-2], block_size=32)
                else:
                    q, scale = simple_mx_quantize(t, axes=[-1], block_size=32)
                save_tensor(q, bin_name(name)) 
                save_tensor(scale, bin_name(name + "_scale")) 
            else:
                # Other tensors are saved as bfloat16
                save_tensor(t.to(torch.bfloat16), bin_name(name))

    # ------------------------------------------
    # 1) QK^T + softmax => skip ONLY if last step
    # ------------------------------------------
    if not on_last_step:
        print("Performing QK^T + softmax...")

        assert Q_tile is not None
        assert K_tile is not None

        qkt_tile = Q_tile @ K_tile.T

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

        # Write .bin files for QK^T+softmax
        save(K_tile,         "k_tile")
        save(max_input,      "max_input")
        save(Q_tile,         "q_tile")
        save(sum_input,      "sum_input")
        save(max_output,     "max_output")
        save(qkt_tile,       "qkt_tile")
        save(rescale_output, "rescale_output")
        save(p_tile,         "softmax_output")
        save(sum_output,     "sum_output")
        
            
        
        # Also write tensors to txt files for human-readable inspection
        if txt or debug:
          with open(os.path.join(step_dir, "k_tile.txt"), 'w') as f:
              f.write(str(K_tile))
          with open(os.path.join(step_dir, "max_input.txt"), 'w') as f:
              f.write(str(max_input))
          with open(os.path.join(step_dir, "q_tile.txt"), 'w') as f:
              f.write(str(Q_tile))
          with open(os.path.join(step_dir, "sum_input.txt"), 'w') as f:
              f.write(str(sum_input))
              
          with open(os.path.join(step_dir, "qkt_tile.txt"), 'w') as f:
              f.write(str(qkt_tile))
          with open(os.path.join(step_dir, "max_output.txt"), 'w') as f:
              f.write(str(max_output))
          with open(os.path.join(step_dir, "rescale_output.txt"), 'w') as f:
              f.write(str(rescale_output))
          with open(os.path.join(step_dir, "softmax_output.txt"), 'w') as f:
              f.write(str(p_tile))
          with open(os.path.join(step_dir, "sum_output.txt"), 'w') as f:
              f.write(str(sum_output))

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

        # Write .bin files for partial p@V
        save(V_tile,         "v_tile")
        save(p_input,        "softmax_input")
        save(o_input,        "o_input")
        save(rescale_input,  "rescale_input")
        save(pv_output,      "pv_output")
        save(o_output,       "o_output")
        
        # Also write tensors to txt files for human-readable inspection
        if txt:
          with open(os.path.join(step_dir, "v_tile.txt"), 'w') as f:
              f.write(str(V_tile))
          with open(os.path.join(step_dir, "softmax_input.txt"), 'w') as f:
              f.write(str(p_input))
          with open(os.path.join(step_dir, "o_input.txt"), 'w') as f:
              f.write(str(o_input))
          with open(os.path.join(step_dir, "rescale_input.txt"), 'w') as f:
              f.write(str(rescale_input))
              
          with open(os.path.join(step_dir, "pv_output.txt"), 'w') as f:
              f.write(str(pv_output))
          with open(os.path.join(step_dir, "o_output.txt"), 'w') as f:
              f.write(str(o_output))
      
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

def run_flash_attention_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    tile_size: int,
    output_dir_root: str,
    dtype: str = "MXINT8",
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
            max_input, sum_input = initialize_stats_worker_1(Q_tile.size(0), Q_tile.size(1))
            p_input, rescale_input, o_input = initialize_stats_worker_2(Q_tile.size(0), Q_tile.size(1))
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
        ) = flash_attention_tile_step(
            global_step=global_step,
            total_steps=total_steps,
            Q_tile=Q_tile_queue[i],
            K_tile=K_tile_queue[i],
            V_tile=V_tile_queue[i],
            p_input=p_input,
            max_input=max_input,
            sum_input=sum_input,
            rescale_input=rescale_input,
            o_input=o_input,
            output_dir_root=output_dir_root,
            dtype=dtype
        )

        sum_queue.append(sum_input)

        if i % num_Q_tiles == 0 and i != 0:
            results_o.append(o_input / sum_queue[i-1])
        
        global_step += 1
        
    # Write model.txt, layers.txt, and tilings.txtpb
    model_txt = os.path.join(output_dir_root, "model.txt")
    write_flash_attention_model(model_txt, total_steps, dtype)

    layers_txt = os.path.join(output_dir_root, "layers.txt")
    write_flash_attention_layers(layers_txt, total_steps)
    
    tilings_txt = os.path.join(output_dir_root, "32x32_1024x1024x1024", "tilings.txtpb")
    os.makedirs(os.path.dirname(tilings_txt), exist_ok=True)
    write_flash_attention_tilings(tilings_txt)
          
    # Combine all partial O blocks
    final_O = torch.cat(results_o, dim=0)
    return final_O

def vanilla_attention(Q, K, V):
    # print("Q @ K.T", Q @ K.T)
    return (Q @ K.T).softmax(dim=-1) @ V

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

def generate_flash_attention_data(context_length=64, output_dir_root="test/compiler/networks/flash_attention/CFLOAT/", random=False, dtype="CFLOAT"):
    print("Generating flash attention gold data.")
    print("Context length:", context_length)

    torch.set_default_dtype(torch.float32 if dtype == "CFLOAT" else torch.bfloat16)
    
    tile_size = 32
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
    
    final_O = run_flash_attention_tiled(Q, K, V, tile_size, output_dir_root, dtype)
    print("Final O shape:", final_O.shape)
    print(final_O)

    final_O_vanilla = vanilla_attention(Q, K, V)
    print("Final O shape:", final_O_vanilla.shape)
    print(final_O_vanilla)

    compare_results(final_O, final_O_vanilla)
    return final_O_vanilla, final_O

if __name__ == "__main__":
    import argparse
    
    def parse_args():
        parser = argparse.ArgumentParser(description='Generate flash attention gold data')
        parser.add_argument('--dtype', type=str, default='CFLOAT', help='Data type (default: cfloat)')
        parser.add_argument('--context_length', type=int, default=64, help='Context length (default: 64)')
        return parser.parse_args()
    
    args = parse_args()

    print(args.dtype)
    if args.dtype not in ['CFLOAT', 'BFLOAT16', 'MXINT8']:
        raise ValueError(f"Invalid data type: {args.dtype}")
    
    generate_flash_attention_data(context_length = args.context_length, output_dir_root = "./flash_attention_output", dtype=args.dtype)