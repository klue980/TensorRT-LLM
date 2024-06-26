import argparse
import json
import os

from tensorrt_llm.quantization.mode import (
    FP8, INT8, W4A8_AWQ, W4A16, W4A16_AWQ, W4A16_GPTQ, W8A8_SQ_PER_CHANNEL,
    W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN, W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
    W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN, W8A8_SQ_PER_TENSOR_PLUGIN, W8A16)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        type=str,
        default='config.json',
        help='The path to save the TensorRT-LLM checkpoint config.json file')
    parser.add_argument('--architecture', type=str, default='GPTForCausalLM')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_key_value_heads', type=int, default=None)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--norm_epsilon', type=float, default=1e-5)
    parser.add_argument('--position_embedding_type',
                        type=str,
                        default='learned_absolute')
    parser.add_argument(
        '--use_parallel_embedding',
        action='store_true',
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--share_embedding_table',
        action='store_true',
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')

    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')

    parser.add_argument('--quant_algo',
                        type=str,
                        default=None,
                        choices=[
                            None, W8A16, W4A16, W4A16_AWQ, W4A8_AWQ, W4A16_GPTQ,
                            W8A8_SQ_PER_CHANNEL, W8A8_SQ_PER_TENSOR_PLUGIN,
                            W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
                            W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN,
                            W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
                        ])
    parser.add_argument('--kv_cache_quant_algo',
                        type=str,
                        default=None,
                        choices=[None, FP8, INT8])
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--has_zero_point', default=False, action='store_true')
    parser.add_argument('--pre_quant_scale', default=False, action='store_true')
    parser.add_argument('--exclude_modules', nargs='+', default=None)

    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--apply_query_key_layer_scaling',
                        default=False,
                        action='store_true')
    parser.add_argument('--rotary_pct', type=float, default=1.0)
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)

    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.')
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help=
        "Add lora in which modules. Only be activated when use_lora_plugin is enabled."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    assert args.output_path.endswith('.json')
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {
        'architecture': args.architecture,
        'dtype': args.dtype,
        'vocab_size': args.vocab_size,
        'max_position_embeddings': args.max_position_embeddings,
        'hidden_size': args.hidden_size,
        'intermediate_size': args.intermediate_size,
        'num_hidden_layers': args.num_hidden_layers,
        'num_attention_heads': args.num_attention_heads,
        'num_key_value_heads': args.num_key_value_heads,
        'hidden_act': args.hidden_act,
        'norm_epsilon': args.norm_epsilon,
        'position_embedding_type': args.position_embedding_type,
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.share_embedding_table,
        'quantization': {
            'quant_algo': args.quant_algo,
            'kv_cache_quant_algo': args.kv_cache_quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'bias': args.bias,
        'apply_query_key_layer_scaling': args.apply_query_key_layer_scaling,
        'rotary_pct': args.rotary_pct,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'max_lora_rank': args.max_lora_rank,
        'lora_target_modules': args.lora_target_modules,
    }

    if args.intermediate_size is None:
        config['intermediate_size'] = args.hidden_size * 4
    if args.num_key_value_heads is None:
        config['num_key_value_heads'] = args.num_attention_heads

    if args.quant_algo is not None:
        if 'AWQ' in args.quant_algo or 'GPTQ' in args.quant_algo:
            config['quantization'].update({
                'group_size':
                args.group_size,
                'has_zero_point':
                args.has_zero_point,
                'pre_quant_scale':
                args.pre_quant_scale,
                'exclude_modules':
                args.exclude_modules,
            })

    with open(args.output_path, 'w') as f:
        json.dump(config, f, indent=4)
