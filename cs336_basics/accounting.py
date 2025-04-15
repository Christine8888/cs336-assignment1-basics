model = "gpt2xl"

vocab_size = 50257

if model == "gpt2xl":
    d_model = 1600
    num_layers = 48
    d_ff = 6400
    num_heads = 25
elif model == "gpt2small":
    d_model = 768
    num_layers = 12
    d_ff = d_model * 4
    num_heads = 12
elif model == "gpt2medium":
    d_model = 1024
    num_layers = 24
    d_ff = d_model * 4
    num_heads = 16
elif model == "gpt2large":
    d_model = 1280
    num_layers = 36
    d_ff = d_model * 4
    num_heads = 20

seq_len = 1024
batch_size = 1


def adamw_accounting():
    # value of the parameter (4 bytes)
    # gradient, 1st moment, 2nd moment (12 bytes)
    # activations: after every layer/operation

    num_bytes = 4 # assume float32
    transformer_kqv_params = 3 * d_model ** 2
    transformer_o_params = d_model ** 2
    transformer_ffn_params = 3 * d_model * 4 * d_model
    transformer_ln_params = 2 * d_model
    transformer_block_params = transformer_kqv_params + transformer_o_params + transformer_ffn_params + transformer_ln_params
    transformer_params = num_layers * transformer_block_params 
    embedding_params = 2 * vocab_size * d_model
    all_params = transformer_params + embedding_params + d_model
    # weight value, gradient, 1st moment, 2nd moment
    adamw_params_memory = num_bytes * 4 * all_params

    print('parameters:', adamw_params_memory)

    # activation accounting
    rmsnorm_acts = 2 * seq_len * d_model
    qkv_acts = 3 * seq_len * d_model
    qtk_acts = seq_len * seq_len * num_heads
    softmax_acts = seq_len * seq_len * num_heads
    value_acts = seq_len * d_model
    o_acts = seq_len * d_model
    ffn_acts = seq_len * (d_ff + d_ff + d_model)
    transformer_block_acts = rmsnorm_acts + qkv_acts + qtk_acts + softmax_acts + value_acts + o_acts + ffn_acts
    final_ln_acts = seq_len * d_model
    final_embedding_acts = seq_len * vocab_size
    logit_acts = seq_len * vocab_size
    all_acts = num_bytes * batch_size * (num_layers * transformer_block_acts + final_ln_acts + final_embedding_acts + logit_acts)
    print('activations:', all_acts)

    return all_params

def adamw_accounting_nonnaive():
    num_bytes = 4 # assume float32

    # start with all activations and adamw state (weight, 1st moment, 2nd moment)
    embedding_acts = seq_len * d_model
    rmsnorm_acts = 2 * seq_len * d_model
    qkv_acts = 3 * seq_len * d_model
    qtk_acts = seq_len * seq_len * num_heads
    softmax_acts = seq_len * seq_len * num_heads
    value_acts = seq_len * d_model
    o_acts = seq_len * d_model
    ffn_acts = seq_len * (d_ff + d_ff + d_model)
    transformer_block_acts = rmsnorm_acts + qkv_acts + qtk_acts + softmax_acts + value_acts + o_acts + ffn_acts
    final_ln_acts = seq_len * d_model
    final_embedding_acts = seq_len * vocab_size
    logit_acts = seq_len * vocab_size
    all_acts = num_bytes * (embedding_acts + num_layers * transformer_block_acts + final_ln_acts + final_embedding_acts + logit_acts)
    print('activation bytes per item:', all_acts)
    all_acts *= batch_size

    transformer_kqv_params = 3 * d_model ** 2
    transformer_o_params = d_model ** 2
    transformer_ffn_params = 3 * d_model * 4 * d_model
    transformer_ln_params = 2 * d_model
    transformer_block_params = transformer_kqv_params + transformer_o_params + transformer_ffn_params + transformer_ln_params
    transformer_params = num_layers * transformer_block_params 
    embedding_params = 2 * vocab_size * d_model
    final_ln_params = d_model
    all_params = num_bytes * (transformer_params + embedding_params + final_ln_params)
    
    starting_memory = all_acts + 3 * all_params

    print('fixed memory:', 3 * all_params)
    print('total starting memory:', starting_memory)

    # final embedding layer
    starting_memory += num_bytes * embedding_params // 2 # gradient through embedding matrix
    starting_memory -= num_bytes * batch_size * logit_acts 
    starting_memory -= num_bytes * batch_size * final_embedding_acts
    print('memory after final embedding:', starting_memory)

    # final ln
    starting_memory += num_bytes * final_ln_params
    starting_memory -= num_bytes * batch_size * final_ln_acts
    print('memory after final ln:', starting_memory)

    # transformer blocks
    starting_memory += num_bytes * transformer_block_params * num_layers
    starting_memory -= num_bytes * batch_size * transformer_block_acts * num_layers
    print('memory after transformer blocks:', starting_memory)

    # embedding layer
    starting_memory += num_bytes * embedding_params // 2
    starting_memory -= num_bytes * batch_size * embedding_acts
    print('memory after embedding:', starting_memory)

    print('expected final memory:', all_params * 4)

    return all_params

def transformer_accounting():
    embedding_matmul = d_model * vocab_size * seq_len

    # 3 W matrices, all d_ff * d_model; residual stream: d_model * seq_len
    transformer_swiglu_matmul = num_layers * 3 * d_model * d_ff * seq_len

    # KQVO matmuls, all d_model * d_model matrices and d_model * seq_len residual stream
    transformer_kqvo_matmul = num_layers * 4 * d_model * d_model * seq_len

    # KQ^T matmul
    transformer_attn_matmul = num_layers * num_heads * seq_len * seq_len * (d_model // num_heads)

    # V matmul
    transformer_value_matmul = num_layers * num_heads * seq_len * seq_len * (d_model // num_heads)

    unembedding_matmul = vocab_size * d_model * seq_len

    transformer_linear = transformer_swiglu_matmul + transformer_kqvo_matmul + embedding_matmul + unembedding_matmul
    transformer_quadratic = transformer_attn_matmul + transformer_value_matmul
    transformer_total = 2 * (transformer_linear + transformer_quadratic)

    #print('embedding matmul: ', embedding_matmul / transformer_total)
    print('transformer total: ', f"{transformer_total}")
    print('transformer ffn matmul: ', 2 * transformer_swiglu_matmul / transformer_total)
    print('transformer kqvo matmul: ', 2 * transformer_kqvo_matmul / transformer_total)
    print('transformer attn matmul: ', 2 * transformer_attn_matmul / transformer_total)
    print('transformer value matmul: ', 2 * transformer_value_matmul / transformer_total)
    print('transformer all attention: ', 2 * (transformer_attn_matmul + transformer_value_matmul + transformer_kqvo_matmul) / transformer_total)
    print('transformer embedding matmuls: ', 2 * (unembedding_matmul + embedding_matmul) / transformer_total)
    # print('unembedding matmul: ', unembedding_matmul / transformer_total)

    # flop accounting
    batch_size = 1024
    forward_flops = transformer_total * batch_size
    backward_flops = transformer_total * 2 * batch_size
    all_params = adamw_accounting()
    adamw_flops = all_params * 14
    total_flops = forward_flops + backward_flops + adamw_flops
    print('flops per iter', f"{total_flops:.2e}")

    gpt_flops = total_flops * 400000
    print('gpt flops', f"{gpt_flops:.2e}")

    a100_flops = 19.5 * 10**12 * 0.5
    a100_cost = gpt_flops / a100_flops
    print('a100 seconds', f"{a100_cost:.2e}")
    print('a100 days', f"{a100_cost / (60 * 60 * 24):.2e}")

if __name__ == "__main__":
    transformer_accounting()