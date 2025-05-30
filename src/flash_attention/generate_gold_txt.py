import os
import textwrap
import re


# ──────────────────────────────  CONSTANTS  ────────────────────────────── #
FLOAT32_BYTES = 4

SIZE_START  = 25216          # bytes used by step 0   (your numbers)
SIZE_STEADY = 62208          # bytes used by steps 1‑n‑2
SIZE_END    = 36992          # bytes used by final step

FIRST_TMPL = """\
inputs {
  node: "q_tile"
  shape: 1
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
  }
}
inputs {
  node: "k_tile"
  shape: 1
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 8192
  }
}
inputs {
  node: "max_input"
  shape: 1
  shape: 32
  dtype: "float32"
  memory {
    address: 16384
  }
}
inputs {
  node: "sum_input"
  shape: 1
  shape: 32
  dtype: "float32"
  memory {
    address: 16512
  }
}
ops {
  op {
    name: "flash_attention"
    op: "call_function"
    target: "flash_attention"
    kwargs {
      key: "query_tile"
      value {
        tensor {
          node: "q_tile"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
          }
        }
      }
    }
    kwargs {
      key: "key_tile"
      value {
        tensor {
          node: "k_tile"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 8192
          }
        }
      }
    }
    kwargs {
      key: "max_input"
      value {
        tensor {
          node: "max_input"
          shape: 1
          shape: 32
          dtype: "float32"
          memory {
            address: 16384
          }
        }
      }
    }
    kwargs {
      key: "sum_input"
      value {
        tensor {
          node: "sum_input"
          shape: 1
          shape: 32
          dtype: "float32"
          memory {
            address: 16512
          }
        }
      }
    }
  }
  outputs {
    tensors {
      node: "qkt_tile"
      shape: 32
      shape: 32
      dtype: "float32"
      memory {
        address: 16640
      }
    }
    tensors {
      node: "softmax_output"
      shape: 32
      shape: 32
      dtype: "float32"
      memory {
        address: 20736
      }
    }
    tensors {
      node: "max_output"
      shape: 32
      dtype: "float32"
      memory {
        address: 24832
      }
    }
    tensors {
      node: "sum_output"
      shape: 32
      dtype: "float32"
      memory {
        address: 24960
      }
    }
    tensors {
      node: "rescale_output"
      shape: 1
      shape: 32
      dtype: "float32"
      memory {
        address: 25088
      }
    }
  }
}
"""

LAST_TMPL = """\
inputs {
  node: "v_tile"
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 0
  }
}
inputs {
  node: "softmax_input"
  shape: 32
  shape: 32
  dtype: "float32"
  memory {
    address: 8192
  }
}
inputs {
  node: "o_input"
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 12288
  }
}
inputs {
  node: "rescale_input"
  shape: 1
  shape: 32
  dtype: "float32"
  memory {
    address: 20480
  }
}
ops {
  op {
    name: "flash_attention"
    op: "call_function"
    target: "flash_attention"
    kwargs {
      key: "value_tile"
      value {
        tensor {
          node: "v_tile"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 0
          }
        }
      }
    }
    kwargs {
      key: "softmax_input"
      value {
        tensor {
          node: "softmax_input"
          shape: 32
          shape: 32
          dtype: "float32"
          memory {
            address: 8192
          }
        }
      }
    }
    kwargs {
      key: "o_input"
      value {
        tensor {
          node: "o_input"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 12288
          }
        }
      }
    }
    kwargs {
      key: "rescale_input"
      value {
        tensor {
          node: "rescale_input"
          shape: 1
          shape: 32
          dtype: "float32"
          memory {
            address: 20480
          }
        }
      }
    }
  }
  outputs {
    tensors {
      node: "pv_output"
      shape: 32
      shape: 64
      dtype: "float32"
      memory {
        address: 20608
      }
    }
    tensors {
      node: "o_output"
      shape: 32
      shape: 64
      dtype: "float32"
      memory {
        address: 28800
      }
    }
  }
}
"""

STEADY_TMPL = """\
inputs {
  node: "q_tile"
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 0
  }
}
inputs {
  node: "k_tile"
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 8192
  }
}
inputs {
  node: "v_tile"
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 16384
  }
}
inputs {
  node: "softmax_input"
  shape: 32
  shape: 32
  dtype: "float32"
  memory {
    address: 24576
  }
}
inputs {
  node: "max_input"
  shape: 1
  shape: 32
  dtype: "float32"
  memory {
    address: 28672
  }
}
inputs {
  node: "sum_input"
  shape: 1
  shape: 32
  dtype: "float32"
  memory {
    address: 28800
  }
}
inputs {
  node: "rescale_input"
  shape: 1
  shape: 32
  dtype: "float32"
  memory {
    address: 28928
  }
}
inputs {
  node: "o_input"
  shape: 32
  shape: 64
  dtype: "float32"
  memory {
    address: 29056
  }
}
ops {
  op {
    name: "flash_attention"
    op: "call_function"
    target: "flash_attention"
    kwargs {
      key: "query_tile"
      value {
        tensor {
          node: "q_tile"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 0
          }
        }
      }
    }
    kwargs {
      key: "key_tile"
      value {
        tensor {
          node: "k_tile"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 8192
          }
        }
      }
    }
    kwargs {
      key: "value_tile"
      value {
        tensor {
          node: "v_tile"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 16384
          }
        }
      }
    }
    kwargs {
      key: "softmax_input"
      value {
        tensor {
          node: "softmax_input"
          shape: 32
          shape: 32
          dtype: "float32"
          memory {
            address: 24576
          }
        }
      }
    }
    kwargs {
      key: "max_input"
      value {
        tensor {
          node: "max_input"
          shape: 1
          shape: 32
          dtype: "float32"
          memory {
            address: 28672
          }
        }
      }
    }
    kwargs {
      key: "sum_input"
      value {
        tensor {
          node: "sum_input"
          shape: 1
          shape: 32
          dtype: "float32"
          memory {
            address: 28800
          }
        }
      }
    }
    kwargs {
      key: "rescale_input"
      value {
        tensor {
          node: "rescale_input"
          shape: 1
          shape: 32
          dtype: "float32"
          memory {
            address: 28928
          }
        }
      }
    }
    kwargs {
      key: "o_input"
      value {
        tensor {
          node: "o_input"
          shape: 32
          shape: 64
          dtype: "float32"
          memory {
            address: 29056
          }
        }
      }
    }
  }
  outputs {
    tensors {
      node: "qkt_tile"
      shape: 32
      shape: 32
      dtype: "float32"
      memory {
        address: 37248
      }
    }
    tensors {
      node: "pv_output"
      shape: 32
      shape: 64
      dtype: "float32"
      memory {
        address: 41344
      }
    }
    tensors {
      node: "o_output"
      shape: 32
      shape: 64
      dtype: "float32"
      memory {
        address: 49536
      }
    }
    tensors {
      node: "softmax_output"
      shape: 32
      shape: 32
      dtype: "float32"
      memory {
        address: 57728
      }
    }
    tensors {
      node: "max_output"
      shape: 32
      dtype: "float32"
      memory {
        address: 61824
      }
    }
    tensors {
      node: "sum_output"
      shape: 32
      dtype: "float32"
      memory {
        address: 61952
      }
    }
    tensors {
      node: "rescale_output"
      shape: 1
      shape: 32
      dtype: "float32"
      memory {
        address: 62080
      }
    }
  }
}
"""

def patch(step: int, base_addr: int, proto: str) -> str:
    """Rename every op/tensor with _step_N and shift all addresses."""
    # rename
    proto = proto.replace('flash_attention"', f'flash_attention_step_{step}"')
    for bare in (
        "q_tile", "k_tile", "v_tile",
        "qkt_tile", "softmax_input", "softmax_output",
        "max_input", "max_output",
        "sum_input", "sum_output",
        "rescale_input", "rescale_output",
        "pv_output", "o_input", "o_output"
    ):
        proto = proto.replace(f'"{bare}"', f'"{bare}_step_{step}"')

    # shift every numeric address literal
    def _shift(match):
        return f"address: {int(match.group(1)) + base_addr}"
    proto = re.sub(r'address:\s+(\d+)', _shift, proto)
    
    return proto


# ───────────────────────────  MAIN ENTRY  ─────────────────────────── #
def write_flash_attention_model(out_txt: str, num_steps: int = 5):
    addr = 0
    with open(out_txt, "w") as fh:
        for s in range(num_steps):
            if s == 0:
                tmpl, size = FIRST_TMPL, SIZE_START
            elif s == num_steps - 1:
                tmpl, size = LAST_TMPL, SIZE_END
            else:
                tmpl, size = STEADY_TMPL, SIZE_STEADY

            fh.write(patch(s, addr, tmpl))
            addr += size
            
if __name__ == "__main__":
    write_flash_attention_model("model.txt")

