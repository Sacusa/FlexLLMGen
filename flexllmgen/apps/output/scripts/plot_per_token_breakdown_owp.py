#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 140]
}
yticks = {
    "opt-175b": 20
}

def gen_plot(load_weight_latency, compute_latency, ideal_load_weight_latency,
             scenarios, model):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    line_axis = axis.twinx()
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = [0, 1]
    width, offset = common.get_width_offset(0.8, len(scenarios))

    weight_means = []
    compute_means = []

    for plot_number, scenario in enumerate(scenarios):
        mean = [np.mean(load_weight_latency[scenario])] * 2
        std = [np.std(load_weight_latency[scenario])] * 2

        weight_means.append(mean[0])

        # bars for load_weight_latency
        axis.bar([i+offset for i in x], mean, edgecolor="black",
                 label=common.scenario_labels[scenario], width=width,
                 color=common.colormap[plot_number], zorder=3.5)
        # for i in x:
        #     axis.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                   elinewidth=4, markerfacecolor="k",
        #                   markeredgecolor="k", zorder=3.5)
        offset += width

    width, offset = common.get_width_offset(0.8, len(scenarios))
    for i in x:
        vals = [np.mean(compute_latency[scenario][i]) for scenario in scenarios]
        line_axis.plot([i + offset + (width*j) for j in range(len(scenarios))],
                       [v-v/2 for v in vals], color="k", linewidth=2,
                        marker="*", mfc="tab:pink", mew=1, mec="k", ms=25,
                        zorder=4.5)
        line_axis.plot([i + offset + (width*j) for j in range(len(scenarios))],
                       [v+v/2 for v in vals], color="k", linewidth=2,
                        marker="D", mfc="tab:pink", mew=1, mec="k", ms=15,
                        zorder=4.5)
        for scenario in scenarios:
            compute_means.append([np.mean(compute_latency[scenario][i]) \
                                  for i in x])

    baseline_index = 1
    for i, scenario in enumerate(scenarios):
        print(scenario, end=" ")
        print(((weight_means[baseline_index] - weight_means[i]) / \
                weight_means[baseline_index]) * 100, end=" ")
        for j in x:
            print(max(compute_means[i][j], weight_means[i]), end=" ")
        for j in x:
            print(compute_means[i][j] / compute_means[baseline_index][j],
                  end=" ")
        for j in x:
            print(compute_means[i][j] / weight_means[i], end=" ")
        print()

    # format x-axis
    axis.set_xlabel("Stage", size=common.font_size["axis_label"])
    axis.set_xticks(x)
    axis.set_xticklabels(["Prefill", "Decode"],
                         size=common.font_size["axis_tick"])

    # format y-axis
    axis.set_ylabel(common.breakdown_labels["load_weight"],
                    size=common.font_size["axis_label"])
    axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    axis.set_ylim(ylim[model])
    axis.yaxis.set_major_locator(plt.MultipleLocator(yticks[model]))

    line_axis.set_ylabel(common.breakdown_labels["compute"],
                         size=common.font_size["axis_label"])
    line_axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    line_axis.set_ylim(ylim[model])
    line_axis.yaxis.set_major_locator(plt.MultipleLocator(yticks[model]))

    # format the plot
    axis.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                ncol=1, mode="expand", borderaxespad=0,
                fontsize=common.font_size["legend"])
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)
    axis.axhline(y=ideal_load_weight_latency, color="k", linestyle="--",
                 linewidth=4, zorder=4.5)

    # save the plot
    plt.savefig(common.plot_dir + f"{model}_per_token_breakdown_owp.pdf",
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

load_weight_latency = {}
compute_latency = {}

warmup_completed = False

for scenario in scenarios:
    if scenario == "O2_owp":
        config = mm_config
    elif scenario == "O3_owp":
        config = dram_config
    else:
        config = nvdram_config
    config_found = False
    warmup_completed = False

    load_weight_latency[scenario] = []
    compute_latency[scenario] = [[], []]

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

        elif line.startswith("load_weight (layer") and warmup_completed:
            load_weight_latency[scenario].append(
                float(line.split(" ")[-2]) * 1000)

        elif line.startswith("compute_layer_prefill  (per-batch)") and \
             warmup_completed:
            compute_latency[scenario][0].append(
                float(line.split(" ")[-2]) * 1000)

        elif line.startswith("compute_layer_decoding (per-batch)") and \
             warmup_completed:
            compute_latency[scenario][1].append(
                float(line.split(" ")[-2]) * 1000)

ideal_weight_load_latency = []

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

gen_plot(load_weight_latency, compute_latency, ideal_weight_load_latency,
         scenarios, model)