def rename_graph_nodes(graph, quantization_scheme, model_name):
    """Renames nodes in the FX graph for better readability, especially for MobileBERT attention.

    Args:
        graph: The torch.fx.Graph object.
        quantization_scheme: The quantization scheme string (e.g., 'int8,qs=microscaling,bs=16').
    """
    # --- Initial Input Renaming ---
    placeholder_mapping = {
        'arg0_1': 'query_tensor',
        'arg1_1': 'key_tensor',
        'arg2_1': 'value_tensor',
        'arg3_1': 'attention_mask',
        'arg4': 'head_mask', 
    }
    for node in graph.nodes:
        if node.op == 'placeholder' and node.name in placeholder_mapping:
            print(f"Renaming placeholder {node.name} to {placeholder_mapping[node.name]}")
            node.name = placeholder_mapping[node.name]
            if node.name in placeholder_mapping:
                node.target = placeholder_mapping[node.target]

    # --- Scheme-Specific Renaming --- 
    if model_name == "MobileBertSelfAttention":
        if "int8,qs=microscaling" in quantization_scheme:
            # This mapping targets the nodes AFTER convert_pt2e and split_multi_head_attention
            # for the microscaling scheme.
            # It renames the *first* matmul of the QK and AV sequences for the first head.
            specific_mapping = {
                # Find the first QK matmul (likely named 'matmul_mx' after decomposition)
                'matmul_mx': 'qk_matmul', 
                'matmul_mx_1': 'qk_matmul_1', 
                'matmul_mx_2': 'qk_matmul_2', 
                'matmul_mx_3': 'qk_matmul_3', 
                # Find the first AV matmul (likely named 'matmul_mx_4' after decomposition and node duplication)
                'matmul_mx_4': 'av_matmul', 
                'matmul_mx_5': 'av_matmul_1', 
                'matmul_mx_6': 'av_matmul_2', 
                'matmul_mx_7': 'av_matmul_3', 
                'linear_mx_default': 'query_proj_mx',
                'linear_mx_default_1': 'key_proj_mx',
                'linear_mx_default_2': 'value_proj_mx',
            }
        elif quantization_scheme == "CFLOAT":
            specific_mapping = {
                'matmul_2': 'qk_matmul', 
                'matmul_3': 'qk_matmul_1', 
                'matmul_4': 'qk_matmul_2', 
                'matmul_5': 'qk_matmul_3', 
                'matmul_6': 'av_matmul', 
                'matmul_7': 'av_matmul_1', 
                'matmul_8': 'av_matmul_2', 
                'matmul_9': 'av_matmul_3', 
                'linear': 'query_proj_mx',
                'linear_1': 'key_proj_mx',
                'linear_2': 'value_proj_mx',
            }
    elif model_name == "SelfAttention":
        if "int8,qs=microscaling" in quantization_scheme:
            specific_mapping = {
                'matmul_mx': 'qk_matmul', 
                'matmul_mx_1': 'av_matmul', 
            }
        elif quantization_scheme == "CFLOAT":
            specific_mapping = {
                'matmul': 'qk_matmul', 
                'matmul_1': 'av_matmul', 
            }
    else:
        raise ValueError(f"Model name {model_name} not supported")

    # Apply the specific mapping
    # We iterate through a copy of nodes because renaming can affect iteration
    nodes_list = list(graph.nodes)
    renamed_nodes = set()
    for node in nodes_list:
        if node.name in specific_mapping and node.name not in renamed_nodes:
            new_name = specific_mapping[node.name]
            # Ensure the new name is unique before assigning
            unique_name = new_name
            count = 1
            while any(n.name == unique_name for n in graph.nodes if n != node):
                unique_name = f"{new_name}_{count}"
                count += 1
            
            print(f"Renaming {node.name} to {unique_name}") 
            node.name = unique_name
            renamed_nodes.add(unique_name) 

    # Relint the graph after renaming to ensure validity
    graph.lint()