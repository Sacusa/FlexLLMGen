#!/usr/bin/python3

"""Complete sentences with FlexLLMGen and OPT models."""
import argparse
import numa
import os
import time

from datasets import load_dataset
from flexgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig,
        str2bool)
from transformers import AutoTokenizer

DEBUG_ENABLED = lambda: args.print_debug_messages == True
DEBUG_PRINT = lambda error_msg : print("[FlexGen] " + error_msg, flush=True) \
    if DEBUG_ENABLED() else None

def print_dram_utilization(numa_nodes):
    DEBUG_PRINT("NUMA Memory Utilization")
    for numa_node in range(numa_nodes):
        utilization = [i for i in numa.memory.node_memory_info(numa_node)][::-1]
        utilization[0] = utilization[1] - utilization[0]
        DEBUG_PRINT("== Node " + str(numa_node) + ": " + \
              "/".join([str(u) for u in utilization]) + \
              " (" + str(round((utilization[0] * 100) / utilization[1], 2)) + \
              "%)")

def main(args):
    num_memory_numa_nodes = numa.info.get_num_configured_nodes()

    # Set NUMA node for memory
    if args.numa_nodes:
        numa_nodes = [int(n) for n in args.numa_nodes.split(",")]
        for numa_node in numa_nodes:
            assert(numa_node < num_memory_numa_nodes)
        numa.memory.set_membind_nodes(*numa_nodes)

    if DEBUG_ENABLED():
        print_dram_utilization(num_memory_numa_nodes)

    # Initialize environment
    env = ExecutionEnv.create(args.offload_dir)

    # Load the dataset
    dataset = load_dataset(os.environ["HF_DATASETS_CACHE"] + "/c4",
        "realnewslike", split="validation")["text"]

    # Offloading policy
    policy = Policy(args.batch_size,  # batch size
                    args.num_batches,  # number of batches
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=args.cpu_cache_compute,
                    attn_sparsity=1.0, compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Load the tokenizer and the model
    DEBUG_PRINT("Initialize")
    tokenizer = AutoTokenizer.from_pretrained(args.model,
        padding_side="left", clean_up_tokenization_spaces=True,
        cache_dir=os.environ["HF_HOME"])
    tokenizer.add_bos_token = False

    model = OptLM(args.model, env, args.path, policy)
    stop = tokenizer("\n").input_ids[0]

    # Generate
    DEBUG_PRINT("Generate")

    if DEBUG_ENABLED():
        print_dram_utilization(num_memory_numa_nodes)

    for prompt_count, prompt in enumerate(dataset):
        if prompt_count == args.num_prompts:
            break

        assert(len(prompt) >= args.input_seq_len)

        tokenized_prompt = tokenizer([prompt[:args.input_seq_len]],
            padding="max_length", max_length=args.input_seq_len).input_ids
        tokenized_prompt = (tokenized_prompt[0],) * args.batch_size * \
            args.num_batches

        for i in range(args.num_generate_repeats):
            start_time = time.time_ns()

            output_ids = model.generate(
                tokenized_prompt,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=args.output_seq_len,
                debug_mode=args.debug_mode,
                stop=stop)

            end_time = time.time_ns()

            outputs = tokenizer.batch_decode(output_ids,
                skip_special_tokens=True)
            num_output_tokens = sum([len(o.split()) for o in outputs])

            if args.print_output_tokens:
                print("Outputs:\n" + 70 * '-', flush=True)
                for i in [0, len(outputs)-1]:
                    print(f"{i}: {outputs[i]}", flush=True)
                    print("-" * 70, flush=True)

            print("Throughput (" + str(prompt_count) + ", " + \
                str(i) + ") = " + str(num_output_tokens) + "/" + \
                str(end_time - start_time) + " tokens/ns", flush=True)

    # Shutdown
    DEBUG_PRINT("Shutdown")
    env.close_copy_threads()

    if DEBUG_ENABLED():
        print_dram_utilization(num_memory_numa_nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str,
        default="/p/nvmgpu/flexgen/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")

    # FlexGen parameters
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
             "the percentage of weight on GPU, "
             "the percentage of weight on CPU, "
             "the percentage of attention cache on GPU, "
             "the percentage of attention cache on CPU, "
             "the percentage of activations on GPU, "
             "the percentage of activations on CPU")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    parser.add_argument("--offload-dir", type=str,
        default="/bigtemp/sgupta45/flexgen/offload_dir",
        help="The directory to offload tensors. ")

    # HuggingFace parameters
    parser.add_argument("--cache-dir", type=str,
        default="/p/nvmgpu/hf_cache",
        help="Cache directory for the model and tokenizer.")
    parser.add_argument("--batch-size", type=int, default=1,
        help="Batch size.")
    parser.add_argument("--num-batches", type=int, default=1,
        help="Number of batches.")
    parser.add_argument("--num-generate-repeats", type=int, default=1,
        help="Number of generate repitions.")
    parser.add_argument("--num-prompts", type=int, default=1,
        help="Number of prompts.")
    parser.add_argument("--input-seq-len", type=int, default=128,
        help="Maximum input sequence length.")
    parser.add_argument("--output-seq-len", type=int, default=256,
        help="Maximum output sequence length.")

    # System parameters
    parser.add_argument("--numa-nodes", type=str, default="",
        help="Comma-separated list of NUMA nodes to allocate the memory on.")
    parser.add_argument("--print-debug-messages", type=str2bool, nargs="?",
        const=True, default=False)
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--print-output-tokens", type=str2bool, nargs="?",
        const=True, default=False)

    args = parser.parse_args()

    assert len(args.percent) == 6

    main(args)
