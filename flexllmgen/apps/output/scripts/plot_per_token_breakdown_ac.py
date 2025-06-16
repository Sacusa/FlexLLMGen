#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 70]
}
yticks = {
    "opt-175b": 10
}

def gen_plot(load_weight_latency, compute_latency, ideal_load_weight_latency,
             scenarios, model, stage):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_medium, dpi=600)
    line_axis = axis.twinx()
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(common.all_cpu_batch_sizes[model])))
    width, offset = common.get_width_offset(0.8, len(scenarios))

    weight_means = []
    compute_means = []

    for plot_number, scenario in enumerate(scenarios):
        mean = []
        std = []
        for i in range(len(load_weight_latency[scenario])):
            mean.append(np.mean(load_weight_latency[scenario][i]))
            std.append(np.std(load_weight_latency[scenario][i]))

        weight_means.append(mean)

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
        line_axis.plot([i + offset + (width*j) for j in range(len(scenarios))],
                       [np.mean(compute_latency[scenario][i]) \
                        for scenario in scenarios], color="k", linewidth=2,
                        marker="*", mfc="tab:pink", mew=1, mec="k", ms=25,
                        zorder=4.5)

        for scenario in scenarios:
            compute_means.append([np.mean(compute_latency[scenario][i]) \
                                  for i in x])

    print(stage)
    baseline_index = 1
    for i, scenario in enumerate(scenarios):
        print(scenario, end=" ")
        for j in x:
            print(((weight_means[baseline_index][j] - weight_means[i][j]) / \
                   weight_means[baseline_index][j]) * 100, end=" ")
        for j in x:
            print(compute_means[i][j] / compute_means[baseline_index][j],
                  end=" ")
        for j in x:
            print(compute_means[i][j] / weight_means[i][j], end=" ")
        print()

    # format x-axis
    axis.set_xlabel("Batch Size", size=common.font_size["axis_label"])
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in common.all_cpu_batch_sizes[model]],
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
    plt.savefig(common.plot_dir + f"{model}_per_token_breakdown_{stage}_ac.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

scenarios = ["O0", "O1_ac", "O2_ac", "O3_ac"]
scenario_dir = {
    "O0"   : "compressed/",
    "O1_ac": "compressed/all_cpu/",
    "O2_ac": "compressed/all_cpu/",
    "O3_ac": "compressed/all_cpu/"
}
nvdram_config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
dram_config = "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
mm_config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"

load_weight_latency = {}
prefill_compute_latency = {}
decode_compute_latency = {}

for scenario in scenarios:
    if scenario == "O2_ac":
        config = mm_config
    elif scenario == "O3_ac":
        config = dram_config
    else:
        config = nvdram_config

    load_weight_latency[scenario] = []
    prefill_compute_latency[scenario] = []
    decode_compute_latency[scenario] = []

    for batch_size in common.all_cpu_batch_sizes[model]:
        if scenario == "O0" and batch_size > 8:
            # DRAM config only supports batch sizes less than or equal to 8
            load_weight_latency[scenario].append([0])
            prefill_compute_latency[scenario].append([0])
            decode_compute_latency[scenario].append([0])
            continue

        load_weight_latency[scenario].append([])
        prefill_compute_latency[scenario].append([])
        decode_compute_latency[scenario].append([])

        config_found = False
        warmup_completed = False

        for line in open(common.output_dir + scenario_dir[scenario] + \
                (f"batch_size_{batch_size}/opt_{model_size}"
                "_exec_time_breakdown.txt")):
            line = line.strip()

            if line.startswith(model):
                if config_found:
                    break
                elif line.startswith(config):
                    config_found = True

            elif line.startswith("Throughput (0, 0)") and config_found:
                warmup_completed = True

            elif line.startswith("load_weight (layer") and warmup_completed:
                load_weight_latency[scenario][-1].append(
                    float(line.split(" ")[-2]) * 1000)

            elif line.startswith("compute_layer_prefill  (per-batch)") and \
                 warmup_completed:
                prefill_compute_latency[scenario][-1].append(
                    float(line.split(" ")[-2]) * 1000)

            elif line.startswith("compute_layer_decoding (per-batch)") and \
                 warmup_completed:
                decode_compute_latency[scenario][-1].append(
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

gen_plot(load_weight_latency, prefill_compute_latency,
         ideal_weight_load_latency, scenarios, model, "prefill")
gen_plot(load_weight_latency, decode_compute_latency,
         ideal_weight_load_latency, scenarios, model, "decode")