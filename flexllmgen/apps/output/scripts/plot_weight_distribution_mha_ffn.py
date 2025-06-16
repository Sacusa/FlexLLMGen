#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

memory_types = ["disk", "cpu", "gpu"]
memory_type_labels = {
    "disk": "Storage",
    "cpu": "Host",
    "gpu": "GPU"
}
colors = {"disk": "r", "cpu": "g", "gpu": "c"}

def plot_layer_wise_distribution(memory_use, layer_size, model, name):
    x = [0, 1]

    plt.figure(figsize=common.figsize_small, dpi=600)
    num_legend_entries = 0

    bottom = [0] * len(x)
    for memory_type in memory_types:
        # vals = [(memory_use[memory_type][i] / layer_size[i]) * 100 for i in x]
        vals = [common.bytes_to_gb(memory_use[memory_type][i]) for i in x]
        if sum(vals) == 0:
            continue

        plt.bar(x, vals, color=colors[memory_type], edgecolor="k",
                bottom=bottom, label=memory_type_labels[memory_type], width=1,
                zorder=3.5)
        for i in x:
            bottom[i] += vals[i]
        num_legend_entries += 1

    plt.xlabel("Layer", size=common.font_size["axis_label"])
    plt.xticks(x,   ["MHA", "FFN"],
               size=common.font_size["axis_tick"])

    plt.ylabel("Weight Size (GB)", size=common.font_size["axis_label"])
    plt.yticks(size=common.font_size["axis_tick"])
    plt.ylim([0, 2.5])

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
            ncol=len(memory_types), mode="expand", borderaxespad=0,
            fontsize=common.font_size["legend"])
    plt.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    plt.savefig(common.plot_dir + \
                f"{model}_{name}_weight_distribution_mha_ffn.pdf",
                bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

per_layer_use = {}
layer_size = {}
configs = []

curr_config = None

for line in open(common.output_dir + f"opt_{model_size}_allocation_trace.txt"):
# for line in open(common.output_dir + "smallest_first_gpu_cpu_disk/" + \
#                  f"opt_{model_size}_allocation_trace.txt"):
    line = line.strip()

    if line.startswith(model):
        curr_config = line.strip()
        configs.append(curr_config)
        per_layer_use[curr_config] = {m: [] for m in memory_types}
        layer_size[curr_config] = []

    elif line.startswith("[FlexGen] Initializing layer"):
        for memory_type in memory_types:
            per_layer_use[curr_config][memory_type].append(0)
        layer_size[curr_config].append(0)

    elif line.startswith("[FlexGen] Allocating weight on"):
        tokens = line.split()
        memory_type = tokens[4][:-1]
        size = int(tokens[-2])

        per_layer_use[curr_config][memory_type][-1] += size
        layer_size[curr_config][-1] += size

for config in configs:
    for memory_type in memory_types:
        per_layer_use[config][memory_type] = \
            per_layer_use[config][memory_type][1:]
    layer_size[config] = layer_size[config][1:]

    plot_layer_wise_distribution(per_layer_use[config], layer_size[config],
                                 model, common.model_config_labels[config])
