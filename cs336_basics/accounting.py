model = "gpt2small"

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

embedding_matmul = 2 * d_model * vocab_size * seq_len

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
