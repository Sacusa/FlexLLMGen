#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

markers = ["o", "s", "D", "^", "v", "p"]
plot_layer_lo = 50
plot_layer_hi = 70

def gen_plot(latency, scenarios, model):
    x = range(len(latency[scenarios[0]]))
    plot_index = 0

    plt.figure(figsize=common.figsize_small, dpi=600)
    for scenario in scenarios:
        plt.plot(x, latency[scenario], color=common.colormap[plot_index],
                 marker=markers[plot_index], mew=1, mec="k", ms=20,
                 label=common.scenario_labels[scenario], linewidth=2,
                 zorder=3.5)
        plot_index += 1

    plt.xlabel("Layer", size=common.font_size["axis_label"])
    plt.ylabel("Latency (ms)", size=common.font_size["axis_label"])

    plt.xticks(list(range(0, plot_layer_hi-plot_layer_lo+1, 10)),
               list(range(plot_layer_lo, plot_layer_hi+1, 10)),
               size=common.font_size["axis_tick"])
    plt.yticks(size=common.font_size["axis_tick"])

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=1, mode="expand", borderaxespad=0,
               fontsize=common.font_size["legend"])
    plt.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    plt.savefig(common.plot_dir + \
                f"{model}_per_layer_load_weight_latency_owp.pdf",
                bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

scenarios = ["O0", "O1_owp", "O2_owp", "O3_owp"]
scenario_dir = {
    "O0"    : "compressed/batch_size_1/",
    "O1_owp": "compressed/mlp_focused/batch_size_1/",
    "O2_owp": "compressed/mlp_focused/batch_size_1/",
    "O3_owp": "compressed/mlp_focused/batch_size_1/"
}
nvdram_config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
dram_config = "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
mm_config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"

latency = {}

for scenario in scenarios:
    if scenario == "O2_owp":
        config = mm_config
    elif scenario == "O3_owp":
        config = dram_config
    else:
        config = nvdram_config
    config_found = False
    warmup_completed = False

    latency[scenario] = []

    for line in open(common.output_dir + scenario_dir[scenario] + \
                     f"opt_{model_size}_exec_time_breakdown.txt"):
        line = line.strip()

        if line.startswith(model):
            if config_found:
                break
            elif line.startswith(config):
                config_found = True

        elif line.startswith("Throughput (0, 0)") and config_found:
            warmup_completed = True

        elif line.startswith("load_weight") and "per-layer" not in line:
            if warmup_completed:
                tokens = line.split()
                layer = int(tokens[2][:-2])

                if len(latency[scenario]) <= layer:
                    latency[scenario].append([])
                assert len(latency[scenario]) > layer

                latency[scenario][layer].append(float(tokens[-2]) * 1000)

    for layer in range(len(latency[scenario])):
        latency[scenario][layer] = np.mean(latency[scenario][layer])
    latency[scenario] = latency[scenario][plot_layer_lo:plot_layer_hi+1]

gen_plot(latency, scenarios, model)