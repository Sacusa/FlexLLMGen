#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

num_layers = {
    "opt-30b": 98,
    "opt-175b": 194
}
ylim = {
    "opt-30b": [0, 18],
    "opt-175b": [0, 140]
}
yticks = {
    "opt-30b": 6,
    "opt-175b": 20
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

    for plot_number, scenario in enumerate(scenarios):
        mean = []
        std = []
        for i in range(len(load_weight_latency[scenario])):
            mean.append(np.mean(load_weight_latency[scenario][i]))
            std.append(np.std(load_weight_latency[scenario][i]))

        # bars for load_weight_latency
        axis.bar([i+offset for i in x], mean, edgecolor="black",
                 label=common.scenario_labels[scenario], width=width,
                 color=common.colormap[plot_number], zorder=3.5)
        for i in x:
            axis.errorbar([i+offset], mean[i], std[i], ecolor="k",
                          elinewidth=4, markerfacecolor="k",
                          markeredgecolor="k", zorder=3.5)
        offset += width

    width, offset = common.get_width_offset(0.8, len(scenarios))
    for i in x:
        mha_compute_latency = {scenario:compute_latency[scenario][i][0::2] \
                               for scenario in scenarios}
        ffn_compute_latency = {scenario:compute_latency[scenario][i][1::2] \
                               for scenario in scenarios}

        line_axis.plot([i + offset + (width*j) for j in range(len(scenarios))],
                       [np.mean(mha_compute_latency[scenario]) \
                        for scenario in scenarios], color="k", linewidth=2,
                        marker="*", mfc="tab:pink", mew=1, mec="k", ms=25,
                        zorder=4.5)
        line_axis.plot([i + offset + (width*j) for j in range(len(scenarios))],
                       [np.mean(ffn_compute_latency[scenario]) \
                        for scenario in scenarios], color="k", linewidth=2,
                        marker="D", mfc="tab:pink", mew=1, mec="k", ms=15,
                        zorder=4.5)

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

scenarios = ["O0", "O1_ac", "O2", "O3", "O4", "O5"]
scenario_dir = {
    "O0": "",
    "O1_ac": "all_cpu/",
    "O2": "compressed/",
    "O3": "compressed/all_cpu/",
    "O4": "compressed/all_cpu/",
    "O5": "compressed/all_cpu/"
}
nvdram_config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
dram_config = "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
mm_config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"

load_weight_latency = {}
prefill_compute_latency = {}
decode_compute_latency = {}

for scenario in scenarios:
    if scenario == "O4":
        config = mm_config
    elif scenario == "O5":
        config = dram_config
    else:
        config = nvdram_config

    load_weight_latency[scenario] = []
    prefill_compute_latency[scenario] = []
    decode_compute_latency[scenario] = []

    for batch_size in common.all_cpu_batch_sizes[model]:
        if scenario in ["O0", "O2"] and \
            batch_size > 8:
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
        is_prefill = True

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

            elif line.startswith("compute_layer (layer") and warmup_completed:
                if "layer 0" in line:
                    continue

                if f"layer {num_layers[model]-1}" in line:
                    is_prefill = False
                    continue

                if is_prefill:
                    prefill_compute_latency[scenario][-1].append(
                        float(line.split(" ")[-2]) * 1000)
                else:
                    decode_compute_latency[scenario][-1].append(
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

gen_plot(load_weight_latency, prefill_compute_latency,
         ideal_weight_load_latency, scenarios, model, "prefill")
gen_plot(load_weight_latency, decode_compute_latency,
         ideal_weight_load_latency, scenarios, model, "decode")