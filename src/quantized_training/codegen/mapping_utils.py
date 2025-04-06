import logging
import operator
import os
import struct

import torch
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

from .param_pb2 import Argument, Memory, OpOverload, Tensor
from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


# Global variable to store the custom save function
_custom_save_function = None

def register_save_function(custom_function):
    """
    Decorator to register a custom function for saving tensors.
    The custom function should accept two arguments: tensor and filename.
    """
    global _custom_save_function
    if not callable(custom_function):
        raise ValueError("The custom function must be callable.")
    _custom_save_function = custom_function
    print(f"Custom save function '{custom_function.__name__}' registered.")
    return custom_function


def _save_tensor(tensor, filename):
    tensor = tensor.float().flatten()
    packed_data = struct.pack(f'{tensor.numel()}f', *tensor.tolist())
    with open(filename, 'wb') as f:
        f.write(packed_data)
    # print(f"Writing tensor to {filename}")


def save_tensor(tensor, filename):
    """
    Save the tensor to a file using the custom save function if defined,
    otherwise use the default _save_tensor function.
    """
    if _custom_save_function is not None:
        _custom_save_function(tensor, filename)
    else:
        _save_tensor(tensor, filename)


def set_tensor_field(field, node, output_dir=None, is_output=False):
    assert isinstance(node, Node) and hasattr(node, 'value'), (
        f"Expected node {node} has value attribute."
    )

    reshape_node = node.meta.get("reshape")
    is_reshape_fused_with_last_op = node.op == "call_module" and not is_output

    if reshape_node is not None and not is_reshape_fused_with_last_op:
        field.reshape.CopyFrom(map_node(reshape_node))
        field.reshape.kwargs["output_shape"].int_list.values.extend(node.shape)
        if not is_output:
            node = reshape_node.args[0]
    elif (getitem_node := node.meta.get("slicing")) is not None:
        field.reshape.CopyFrom(map_node(getitem_node))
        field.reshape.kwargs["output_shape"].int_list.values.extend(node.shape)
        node = getitem_node.args[0]

    if (dq_scale := node.meta.get("dq_scale", None)) is not None:
        field.scale = dq_scale
        node = node.args[0]

    if (source_node := node.meta.get("source_node", None)) is not None:
        node = source_node

    field.node = node.name
    field.shape.extend(list(node.shape) or [1])

    if (dtype := node.meta.get("dtype", None)) is not None:
        field.dtype = dtype
    else:
        field.dtype = str(node.value.dtype).split(".")[1]

    if (memory := node.meta.get("memory", None)) is not None:
        field.memory.partition = memory.partition_id
        field.memory.address = memory.start

    if output_dir is not None:
        save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))


def set_output_field(param, node, output_dir):
    if isinstance(node.value, torch.Tensor):
         set_tensor_field(param.output, node, output_dir, True)
    elif isinstance(node.value, (tuple, list)):
        partition = node.meta["memory"].partition_id
        address = node.meta["memory"].start

        dtypes = [str(t.dtype).split(".")[1] for t in node.value]
        if "dtype" in node.meta:
            dtypes = [dt or dtypes[i] for i, dt in enumerate(node.meta["dtype"])]

        for i, tensor in enumerate(node.value):
            param.outputs.tensors.append(Tensor(
                node=f"{node.name}_{i}",
                shape=list(tensor.shape),
                dtype=dtypes[i],
                memory=Memory(partition=partition, address=int(address)),
            ))

            address += tensor.numel() * dtype_byte_size(dtypes[i])

            if output_dir is not None:
                save_tensor(tensor, os.path.join(output_dir, f"{node.name}_{i}.bin"))
    else:
        logger.warning(f"Unsupported output type: {type(node.value)}")


def convert_arg(value) -> Argument:
    """
    Converts an argument (which could be a Tensor, list, int, float, etc.)
    into an Argument protobuf.
    """
    arg = Argument()

    if isinstance(value, torch.fx.Node):
        if isinstance(value.value, torch.Tensor):
            set_tensor_field(arg.tensor, value)
        elif isinstance(value.value, (tuple, list)):
            arg.tensor_list.tensors.extend([
                Tensor(node=f"{value.name}_{i}") for i in range(len(value.value))
            ])
    elif isinstance(value, int):
        arg.int_value = value
    elif isinstance(value, float):
        arg.float_value = value
    elif isinstance(value, bool):
        arg.bool_value = value
    elif isinstance(value, str):
        arg.str_value = value
    elif isinstance(value, (
        torch.dtype, torch.layout, torch.device, torch.memory_format
    )):
        arg.str_value = str(value).split(".")[-1]
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, torch.fx.Node) or x is None for x in value):
            arg.tensor_list.tensors.extend([
                convert_arg(x).tensor if x is not None else Tensor(is_none=True)
                for x in value
            ])
        elif all(isinstance(x, int) for x in value):
            arg.int_list.values.extend(value)
        elif all(isinstance(x, bool) for x in value):
            arg.bool_list.values.extend(value)
        elif all(isinstance(x, (int, float, bool)) for x in value):
            arg.scalar_list.values.extend(value)
        else:
            raise TypeError(f"Unsupported list value: {value}")
    else:
        raise TypeError(f"Unsupported arg type: {type(value)}")

    return arg


def map_node(node: torch.fx.Node, output_dir=None) -> OpOverload:
    """
    Converts a torch.fx.Node into an OpOverload protobuf message.
    """
    if hasattr(node.target, "_schema"):
        target = node.target._schema.name.split('::')[1]
    else:
        target = str(node.target)

    op_overload = OpOverload(
        name=node.name,
        op=node.op,
        target=target,
    )

    if _is_nop(node) or is_addressing_op(node) or node.target == operator.getitem:
        op_overload.op = "nop"

    new_args_and_kwargs = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )

    if new_args_and_kwargs is not None:
        args, kwargs = new_args_and_kwargs.args, new_args_and_kwargs.kwargs
    else:
        args, kwargs = node.args, node.kwargs

    # Convert positional arguments
    for arg in args:
        op_overload.args.append(convert_arg(arg))

    # Convert keyword arguments
    for key, value in kwargs.items():
        if "code" not in key and value is not None:
            op_overload.kwargs[key].CopyFrom(convert_arg(value))

    return op_overload


def _is_gemm_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.conv2d_mx.default,
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def _is_elementwise_op(node: Node) -> bool:
    return node.target in [
        # Core aten ops
        torch.ops.aten.abs.default,
        torch.ops.aten.acos.default,
        torch.ops.aten.acosh.default,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.asin.default,
        torch.ops.aten.asinh.default,
        torch.ops.aten.atan.default,
        torch.ops.aten.atan2.default,
        torch.ops.aten.atan2.out,
        torch.ops.aten.atanh.default,
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Scalar,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.ceil.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp.Tensor,
        torch.ops.aten.cos.default,
        torch.ops.aten.cosh.default,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.div.Scalar_mode,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.erf.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.expm1.default,
        torch.ops.aten.floor.default,
        torch.ops.aten.fmod.Scalar,
        torch.ops.aten.fmod.Tensor,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gelu.default,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.isinf.default,
        torch.ops.aten.isnan.default,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.log.default,
        torch.ops.aten.log10.default,
        torch.ops.aten.log1p.default,
        torch.ops.aten.log2.default,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_not.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_xor.default,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.nonzero.default,
        torch.ops.aten.pow.Scalar,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.reciprocal.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.remainder.Scalar,
        torch.ops.aten.remainder.Tensor,
        torch.ops.aten.round.default,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.sign.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.sinh.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.tan.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.trunc.default,
        torch.ops.aten.where.self,
        # Inplace versions of the above operations
        torch.ops.aten.add_.Scalar,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.mul_.Scalar,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.gelu_.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.relu_.default,
        # Quantization operations
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.vmap.default,
        # Not in the core aten operator set. Will be removed in the future.
        torch.ops.aten.silu.default,
    ]


def _is_reshape_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ]


def _is_indexing_or_concatenation_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.stack.default,
        torch.ops.aten.cat.default,
    ]


def _is_nop(node: Node) -> bool:
    """
    The following operations do not require any computation nor handling
    on the memory placement side. Generate a NOP instruction for these ops
    to keep the compute graph intact.
    """
    # A slice from 0 to the end of the input tensor
    if node.target == torch.ops.aten.slice.Tensor:
        input_shape = node.args[0].shape
        default_args = [0, None, None, 1]
        dim, start, end, step = list(node.args[1:]) + default_args[len(node.args) - 1:]
        return start == 0 and end > input_shape[dim] and step == 1

    return node.target in [
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.copy_.default,
        torch.ops.aten.dropout.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    ]


def is_addressing_op(node: Node) -> bool:
    """
    The following operations are handled by the memory placement and
    thus require no additional handling:
    """
    if node.target == torch.ops.aten.select.int:
        return all(d == 1 for d in node.args[0].value.shape[:node.args[1]])

    if node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
        return len(node.args) == 1 or node.args[1] == 0

    return False
