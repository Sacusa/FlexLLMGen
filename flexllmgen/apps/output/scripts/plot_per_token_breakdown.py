#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

load_weight_ylim = {
    "opt-30b": [0, 16],
    "opt-175b": [0, 350]
}
compute_ylim = {
    "opt-30b": [0, 16],
    "opt-175b": [0, 14]
}
num_yticks = {
    "opt-30b": 8,
    "opt-175b": 7
}

def gen_plot(load_weight_latency, compute_latency, ideal_load_weight_latency,
             configs, model, stage):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    line_axis = axis.twinx()
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(common.batch_sizes[model])))
    width, offset = common.get_width_offset(0.8, len(configs))

    for plot_number, config in enumerate(configs):
        mean = []
        std = []
        print(common.model_config_labels[config], end=" ")
        for i in range(len(load_weight_latency[config])):
            mean.append(np.mean(load_weight_latency[config][i]))
            std.append(np.std(load_weight_latency[config][i]))
            print(mean[i], ((mean[i] - ideal_load_weight_latency) / \
                   ideal_load_weight_latency) * 100, end=" ")
        print()

        # bars for load_weight_latency
        axis.bar([i+offset for i in x], mean, edgecolor="black",
                 label=common.model_config_labels[config], width=width,
                 color=common.colormap[plot_number], zorder=3.5)
        # for i in x:
        #     axis.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                   elinewidth=4, markerfacecolor="k",
        #                   markeredgecolor="k", zorder=3.5)
        offset += width

    width, offset = common.get_width_offset(0.8, len(configs))
    for i in x:
        line_axis.plot([i + offset + (width * j) for j in range(len(configs))],
                       [np.mean(compute_latency[config][i]) \
                        for config in configs], color="k", linewidth=2,
                        marker="*", mfc="tab:pink", mew=1, mec="k", ms=25,
                        zorder=4.5)

    # format x-axis
    axis.set_xlabel("Batch Size", size=30)
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in common.batch_sizes[model]],
                         size=common.font_size["axis_tick"])

    # format y-axis
    axis.set_ylabel(common.breakdown_labels["load_weight"],
                    size=common.font_size["axis_label"])
    axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    axis.set_ylim(load_weight_ylim[model])
    axis.yaxis.set_major_locator(plt.MultipleLocator(
        load_weight_ylim[model][1] / num_yticks[model]))

    line_axis.set_ylabel(common.breakdown_labels["compute"],
                         size=common.font_size["axis_label"])
    line_axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    line_axis.set_ylim(compute_ylim[model])
    line_axis.yaxis.set_major_locator(plt.MultipleLocator(
        compute_ylim[model][1] / num_yticks[model]))

    # format the plot
    axis.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                ncol=2, mode="expand", borderaxespad=0,
                fontsize=common.font_size["legend"])
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    axis.axhline(y=ideal_load_weight_latency, color="k", linestyle="--",
                 linewidth=4, zorder=4.5)

    # save the plot
    plt.savefig(common.plot_dir + f"{model}_per_token_breakdown_{stage}.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

load_weight_latency = {}
prefill_compute_latency = {}
decode_compute_latency = {}
configs = []

config = None
warmup_completed = False

for batch_size in common.batch_sizes[model]:
    for line in open(common.output_dir + \
            (f"batch_size_{batch_size}/opt_{model_size}"
             "_exec_time_breakdown.txt")):
        line = line.strip()

        if line.startswith(model):
            config = line.strip()
            if config not in configs:
                load_weight_latency[config] = [[]]
                prefill_compute_latency[config] = [[]]
                decode_compute_latency[config] = [[]]
                configs.append(config)
            else:
                load_weight_latency[config].append([])
                prefill_compute_latency[config].append([])
                decode_compute_latency[config].append([])
            warmup_completed = False

        elif line.startswith("Throughput (0, 0)"):
            warmup_completed = True

        # elif line.startswith("load_weight            (per-layer)"):
        elif line.startswith("load_weight (layer"):
            if warmup_completed:
                load_weight_latency[config][-1].append(
                    float(line.split(" ")[-2]) * 1000)

        elif line.startswith("compute_layer_prefill  (per-batch)"):
            if warmup_completed:
                prefill_compute_latency[config][-1].append(
                    float(line.split(" ")[-2]) * 1000)

        elif line.startswith("compute_layer_decoding (per-batch)"):
            if warmup_completed:
                decode_compute_latency[config][-1].append(
                    float(line.split(" ")[-2]) * 1000)

ideal_weight_load_latency = []
if model == "opt-30b":
    ideal_weight_load_latency = -1
elif model == "opt-175b":
    config_found = False
    warmup_completed = False

    for line in open(common.output_dir + \
                     "hidden_layers_8/opt_175b_exec_time_breakdown.txt"):
        line = line.strip()

        if line.startswith("opt-175b"):
            if config_found:
                break
            elif line.startswith("opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"):
                config_found = True

        elif line.startswith("Throughput (0, 0)") and config_found:
            warmup_completed = True

        elif line.startswith("load_weight            (per-layer)") and \
             warmup_completed:
            ideal_weight_load_latency.append(float(line.split(" ")[-2]) * 1000)

    ideal_weight_load_latency = np.mean(ideal_weight_load_latency)

# for config in configs:
#     print(common.model_config_labels[config],
#           np.mean(prefill_compute_latency[config][1]) / \
#           np.mean(prefill_compute_latency[config][0]))

gen_plot(load_weight_latency, prefill_compute_latency,
         ideal_weight_load_latency, configs, model, "prefill")
gen_plot(load_weight_latency, decode_compute_latency,
         ideal_weight_load_latency, configs, model, "decode")