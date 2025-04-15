model = "gpt2xl"

vocab_size = 50257
context_size = 1024

if model == "gpt2xl":
    d_model = 1600
    num_layers = 48
    d_ff = 6400
elif model == "gpt2small":
    d_model = 768
    num_layers = 12
    d_ff = 3072
elif model == "gpt2medium":
    d_model = 1024
    num_layers = 24
    d_ff = 4096
elif model == "gpt2large":
    d_model = 1280
    num_layers = 36
    d_ff = 5120

seq_len = 1024
batch_size = 1


def adamw_accounting():
    # value of the parameter (4 bytes)
    # gradient, 1st moment, 2nd moment (12 bytes)
    # activations: after every layer/operation
    num_bytes = 4
    transformer_block_params = 4 * d_model ** 2 + 3 * d_model * 4 * d_model + 2 * d_model
    transformer_params = num_layers * transformer_block_params 
    embedding_params = 2 * vocab_size * d_model
    all_params = transformer_params + embedding_params + d_model

    # weight value, gradient, 1st moment, 2nd moment
    adamw_params_memory = num_bytes * 4 * all_params

    print('parameters:', adamw_params_memory)

    # activation accounting
    rmsnorm_acts = 2 * batch_size * seq_len * d_model
    qkv_acts = 3 * batch_size * seq_len * d_model
    qtk_acts = batch_size * seq_len * seq_len
    softmax_acts = batch_size * seq_len * seq_len
    value_acts = batch_size * seq_len * d_model
    output_acts = batch_size * seq_len * d_model
    ffn_acts = batch_size * seq_len * (d_ff + d_ff + d_model)
    block_acts = rmsnorm_acts + qkv_acts + qtk_acts + softmax_acts + value_acts + output_acts + ffn_acts

    final_ln_acts = batch_size * seq_len * d_model
    final_embedding_acts = batch_size * seq_len * vocab_size
    logit_acts = batch_size * seq_len * vocab_size

    all_acts = num_bytes * (num_layers * block_acts + final_ln_acts + final_embedding_acts + logit_acts)
    print('activations:', all_acts)

def transformer_accounting():
    embedding_matmul = 2 * d_model * vocab_size * seq_len

    # 3 W matrices, all d_ff * d_model; residual stream: d_model * seq_len
    transformer_swiglu_matmul = num_layers * 3 * 2 * d_model * d_ff * seq_len

    transformer_kqvo_matmul = num_layers * 4 * 2 * d_model * d_model * seq_len

    transformer_attn_matmul = num_layers * 2 * seq_len * seq_len * d_model

    transformer_value_matmul = num_layers * 2 * seq_len * seq_len * d_model

    unembedding_matmul = 2 * vocab_size * d_model * seq_len

    transformer_linear = transformer_swiglu_matmul + transformer_kqvo_matmul + embedding_matmul + unembedding_matmul
    transformer_quadratic = transformer_attn_matmul + transformer_value_matmul
    transformer_total = transformer_linear + transformer_quadratic

    #print('embedding matmul: ', embedding_matmul / transformer_total)
    print('transformer total: ', f"{transformer_total:.2e}")
    print('transformer ffn matmul: ', transformer_swiglu_matmul / transformer_total)
    print('transformer kqvo matmul: ', transformer_kqvo_matmul / transformer_total)
    print('transformer attn matmul: ', transformer_attn_matmul / transformer_total)
    print('transformer value matmul: ', transformer_value_matmul / transformer_total)
    # print('transformer attention matmuls: ', (transformer_kqvo_matmul + transformer_attn_matmul + transformer_value_matmul) / transformer_total)
    print('transformer embedding matmuls: ', (unembedding_matmul + embedding_matmul) / transformer_total)
    # print('unembedding matmul: ', unembedding_matmul / transformer_total)

    total_linear = embedding_matmul + transformer_linear + unembedding_matmul
    total_quadratic = transformer_quadratic

    # print('total linear: ', total_linear)
    # print('total quadratic: ', total_quadratic)

if __name__ == "__main__":
    adamw_accounting()