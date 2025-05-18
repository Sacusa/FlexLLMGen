#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

labels = {
    "opt-175b,ssd/storage,nvm/na,dram/memory,gpu/dma": "ssd",
    "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma": "nvm"
}

colors = ["r", "g", "c", "m"]
markers = ["o", "s", "D", "^"]

memory_types = ["disk", "cpu", "gpu"]

def plot_layer_wise_distribution(memory_use, layer_size, model_size, name):
    x = range(len(layer_size))
    plot_index = 0

    plt.figure(figsize=(24, 6), dpi=600)

    bottom = [0] * len(x)
    for memory_type in memory_types:
        # plt.plot(x, [(memory_use[memory_type][i] / layer_size[i]) * 100
        #              for i in x],
        #          color=colors[plot_index], marker=markers[plot_index],
        #          label=memory_type, linewidth=3, zorder=3.5)
        vals = [(memory_use[memory_type][i] / layer_size[i]) * 100 for i in x]
        plt.bar(x, vals, color=colors[plot_index], edgecolor="k", bottom=bottom,
                 label=memory_type, width=1, zorder=3.5)
        for i in x:
            bottom[i] += vals[i]
        plot_index += 1

    plt.xlabel("Layer", size=30)
    plt.ylabel("Percent weights (%)", size=30)

    # plt.xticks(x, size=25)
    plt.yticks(size=25)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=len(memory_types), mode="expand", borderaxespad=0,
               fontsize=25)
    plt.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    plt.savefig(common.plot_dir + \
        f"opt_{model_size}_{name}_layer_wise_weight_distribution.pdf",
        bbox_inches="tight")

def plot_memory_type_bins(memory_use, model_size, name):
    bins = [0] + [2**i for i in range(0, 13)]
    bin_values = {}
    for memory_type in memory_types:
        bin_values[memory_type] = np.histogram([common.bytes_to_mb(s) \
                for s in memory_use[memory_type]], bins=bins)[0]

    x = range(len(bins) - 1)
    width = 0.25
    offset = 0.25
    plot_index = 0

    plt.figure(figsize=(24, 6), dpi=600)
    for memory_type in memory_types:
        plt.bar([i + offset for i in x], bin_values[memory_type],
                width=width, color=colors[plot_index], label=memory_type,
                edgecolor="k", zorder=3.5)
        offset += width
        plot_index += 1

    plt.xlabel("Weight size (MB)", size=30)
    plt.ylabel("Number of weights", size=30)

    plt.margins(x=0)
    plt.xticks(range(len(bins)), [str(b) for b in bins], size=25)
    plt.yticks(size=25)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=len(memory_types), mode="expand", borderaxespad=0,
               fontsize=25)
    plt.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    plt.savefig(common.plot_dir + \
        f"opt_{model_size}_{name}_memory_type_bins.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]

per_layer_use = {}
per_weight_use = {}
layer_size = {}
configs = []

curr_config = None

size_count = {}

for line in open(common.output_dir + f"opt_{model_size}_allocation_trace.txt"):
# for line in open(common.output_dir + "smallest_first_gpu_cpu_disk/" + \
#                  f"opt_{model_size}_allocation_trace.txt"):
    line = line.strip()

    if line.startswith("opt-175b"):
        curr_config = line.strip()
        configs.append(curr_config)
        per_layer_use[curr_config] = {m: [] for m in memory_types}
        per_weight_use[curr_config] = {m: [] for m in memory_types}
        layer_size[curr_config] = []

        if len(size_count) > 0:
            print("Size count:")
            for size in sorted(size_count.keys()):
                print(f"{size}: {size_count[size]} ({common.bytes_to_mb(size * size_count[size])})")
            size_count = {}

    elif line.startswith("[FlexGen] Initializing layer"):
        for memory_type in memory_types:
            per_layer_use[curr_config][memory_type].append(0)
        layer_size[curr_config].append(0)

    elif line.startswith("[FlexGen] Allocating weight on"):
        tokens = line.split()
        memory_type = tokens[4][:-1]
        size = int(tokens[-2])

        per_layer_use[curr_config][memory_type][-1] += size
        per_weight_use[curr_config][memory_type].append(size)
        layer_size[curr_config][-1] += size
        size_count[size] = size_count.get(size, 0) + 1

bins = [2**i for i in range(0, 13)]

for config in configs:
    plot_layer_wise_distribution(per_layer_use[config], layer_size[config],
                                 model_size, labels[config])
    plot_memory_type_bins(per_weight_use[config], model_size, labels[config])

    total_size = 0
    for memory_type in memory_types:
        total_size += sum(per_layer_use[config][memory_type])
    print(config)
    for memory_type in memory_types:
        print(memory_type,
              common.bytes_to_mb(sum(per_layer_use[config][memory_type])))
