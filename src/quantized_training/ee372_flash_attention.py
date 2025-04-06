def rename_graph_nodes(graph):
    # Create a mapping of old names to new names
    name_mapping = {
        # Input placeholders
        'arg0_1': 'hidden_states',
        'arg1_1': 'attention_mask',
        'arg2': 'head_mask',
        
        # Bottleneck input transformation
        '_param_constant0': 'bottleneck_input_weight',
        '_param_constant1': 'bottleneck_input_bias',
        'linear': 'bottleneck_input_proj',
        '_param_constant2': 'bottleneck_input_norm_weight',
        'mul': 'bottleneck_input_norm_scale',
        '_param_constant3': 'bottleneck_input_norm_bias',
        'add': 'bottleneck_input_norm',
        
        # Bottleneck attention transformation
        '_param_constant4': 'bottleneck_attn_weight',
        '_param_constant5': 'bottleneck_attn_bias',
        'linear_1': 'bottleneck_attn_proj',
        '_param_constant6': 'bottleneck_attn_norm_weight',
        'mul_1': 'bottleneck_attn_norm_scale',
        '_param_constant7': 'bottleneck_attn_norm_bias',
        'add_1': 'bottleneck_attn_norm',
        
        # Query projection
        '_param_constant8': 'query_weight',
        '_param_constant9': 'query_bias',
        'linear_2': 'query_proj',
        
        # Key projection
        '_param_constant10': 'key_weight',
        '_param_constant11': 'key_bias',
        'linear_3': 'key_proj',
        
        # Value projection
        '_param_constant12': 'value_weight',
        '_param_constant13': 'value_bias',
        'linear_4': 'value_proj',
        
        # Attention computation
        'matmul': 'qk_matmul',
        'div': 'attn_scaling',
        'add_2': 'attn_mask_add',
        'softmax': 'attn_softmax',
        'dropout': 'attn_dropout',
        'mul_2': 'head_mask_mul',
        'matmul_1': 'av_matmul',
        'permute': 'context_permute',
        'view': 'context_reshape'
    }
    
    # Rename each node
    for node in graph.nodes:
        if node.name in name_mapping:
            node.name = name_mapping[node.name]
