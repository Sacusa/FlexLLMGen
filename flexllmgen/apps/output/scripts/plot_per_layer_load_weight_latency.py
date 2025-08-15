#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

markers = ["o", "s", "D", "^"]
num_plot_layers = 70
xlim = {
    "opt-30b": min(101, num_plot_layers + 1),
    "opt-175b": min(201, num_plot_layers + 1)
}

def gen_plot(latency, configs, model):
    x = range(len(latency[configs[0]]))
    plot_index = 0

    plt.figure(figsize=common.figsize_large, dpi=600)
    for config in configs:
        plt.plot(x, latency[config], color=common.model_config_colors[config],
                 marker=markers[plot_index], mew=1, mec="k", ms=20,
                 label=common.model_config_labels[config], linewidth=2,
                 zorder=3.5)
        plot_index += 1

    plt.xlabel("Layer", size=common.font_size["axis_label"])
    plt.ylabel("Latency (ms)", size=common.font_size["axis_label"])

    plt.xticks(list(range(0, xlim[model], 10)),
               size=common.font_size["axis_tick"])
    plt.yticks(size=common.font_size["axis_tick"])

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=len(configs), mode="expand", borderaxespad=0,
               fontsize=common.font_size["legend"])
    plt.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    plt.savefig(common.plot_dir + f"{model}_per_layer_load_weight_latency.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

latency = {}
configs = []

config = None
warmup_completed = False

for line in open(common.output_dir + \
        (f"batch_size_1/opt_{model_size}_exec_time_breakdown.txt")):
    line = line.strip()

    if line.startswith(model):
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

            latency[config][layer].append(float(tokens[-2]) * 1000)

for config in configs:
    for layer in range(len(latency[config])):
        latency[config][layer] = np.mean(latency[config][layer])
    latency[config] = latency[config][:num_plot_layers]

gen_plot(latency, configs, model)