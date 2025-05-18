#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

labels = {
    "opt-175b,ssd/storage,nvm/na,dram/memory,gpu/dma": "SSD",
    "opt-175b,ssd/na,nvm/storage,dram/memory,gpu/dma": "FSDAX",
    "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma": "NVDRAM",
    "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma": "MemoryMode"
}

def gen_plot(load_weight_latency, compute_latency, configs, model_size,
             batch_size):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=(18, 6), dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(configs)))
    width = 0.4

    plt.bar([i - width/2 for i in x],
            [np.mean(load_weight_latency[config]) for config in configs],
            width=width, edgecolor="black", zorder=3.5, label="Load weight")
    plt.bar([i + width/2 for i in x],
            [np.mean(compute_latency[config]) for config in configs],
            width=width, edgecolor="black", zorder=3.5, label="Compute")
    print([np.mean(compute_latency[config]) for config in configs])
    
    # format x-axis
    axis.set_xticks(x)
    axis.set_xticklabels([labels[c] for c in configs], size=25)

    # format y-axis
    axis.yaxis.set_tick_params(labelsize=25)
    axis.set_ylabel("Latency (s)", size=30)
    axis.set_ylim([0, 0.32])
    axis.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    # format the plot
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=2, mode="expand", borderaxespad=0, fontsize=25)
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)

    # save the plot
    plt.savefig(common.plot_dir + \
        f"opt_{model_size}_batch_size_{batch_size}_per_token_breakdown.png",
        bbox_inches="tight")

if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <model size> <batch size>")
    sys.exit(1)

model_size = sys.argv[1]
batch_size = int(sys.argv[2])

load_weight_latency = {}
compute_latency = {}
configs = []

config = None
warmup_completed = False

for line in open(common.output_dir + \
        f"batch_size_{batch_size}/opt_{model_size}_exec_time_breakdown.txt"):
    line = line.strip()

    if line.startswith(f"opt-{model_size}"):
        config = line.strip()
        load_weight_latency[config] = []
        compute_latency[config] = []
        configs.append(config)
        warmup_completed = False
    
    elif line.startswith("Throughput (0, 0)"):
        warmup_completed = True
    
    elif line.startswith("load_weight            (per-layer)"):
        if warmup_completed:
            load_weight_latency[config].append(float(line.split(" ")[-2]))
    
    elif line.startswith("compute_layer_decoding (per-batch)"):
        if warmup_completed:
            compute_latency[config].append(float(line.split(" ")[-2]))

gen_plot(load_weight_latency, compute_latency, configs, model_size, batch_size)