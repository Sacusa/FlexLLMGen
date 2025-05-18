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

colors = ["r", "g", "b", "m"]
markers = ["o", "s", "D", "^"]

def gen_plot(latency, configs, model_size, batch_size):
    x = range(len(latency[configs[0]]))
    plot_index = 0

    plt.figure(figsize=(24, 6), dpi=600)
    for config in configs:
        plt.plot(x, latency[config], color=colors[plot_index],
                 marker=markers[plot_index], label=labels[config], linewidth=3,
                 zorder=3.5)
        plot_index += 1

    plt.xlabel("Layer", size=30)
    plt.ylabel("Latency (s)", size=30)

    # plt.xticks(x, size=25)
    plt.yticks(size=25)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=len(configs), mode="expand", borderaxespad=0, fontsize=25)
    plt.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    plt.savefig(common.plot_dir + \
        (f"opt_{model_size}_batch_size_{batch_size}"
         "_per_layer_load_weight_latency.pdf"),
        bbox_inches="tight")

if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <model size> <batch size>")
    sys.exit(1)

model_size = sys.argv[1]
batch_size = int(sys.argv[2])

latency = {}
configs = []

config = None
warmup_completed = False

for line in open(common.output_dir + \
        f"batch_size_{batch_size}/opt_{model_size}_exec_time_breakdown.txt"):
    line = line.strip()

    if line.startswith(f"opt-{model_size}"):
        config = line.strip()
        latency[config] = []
        configs.append(config)
        warmup_completed = False

    elif line.startswith("Throughput (0, 0)"):
        warmup_completed = True

    elif line.startswith("load_weight") and "per-layer" not in line:
        if warmup_completed:
            tokens = line.split()
            layer = int(tokens[2][:-2])

            if len(latency[config]) <= layer:
                latency[config].append([])
            assert len(latency[config]) > layer

            latency[config][layer].append(float(tokens[-2]))

for config in configs:
    for layer in range(len(latency[config])):
        latency[config][layer] = np.mean(latency[config][layer])

gen_plot(latency, configs, model_size, batch_size)