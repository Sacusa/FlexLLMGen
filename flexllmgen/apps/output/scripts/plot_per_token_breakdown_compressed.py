#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 90]
}
yticks = {
    "opt-175b": 15
}

def gen_plot(load_weight_latency, compute_latency, scenarios, model, stage):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(common.batch_sizes[model])))
    line_axis = axis.twinx()
    width, offset = common.get_width_offset(0.8, len(scenarios))

    load_means = []
    compute_means = []

    for scenario in scenarios:
        mean = []
        std = []
        for i in range(len(load_weight_latency[scenario])):
            mean.append(np.mean(load_weight_latency[scenario][i]))
            std.append(np.std(load_weight_latency[scenario][i]))

        load_means.append(mean)

        # bars for load_weight_latency
        axis.bar([i+offset for i in x], mean, edgecolor="black",
                 label=scenario, width=width,
                 color=colormap[scenario], zorder=3.5)
        # for i in x:
        #     axis.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                   elinewidth=4, markerfacecolor="k",
        #                   markeredgecolor="k", zorder=3.5)
        offset += width

    width, offset = common.get_width_offset(0.8, len(scenarios))
    for i in x:
        line_axis.plot([i + offset + (width * j) for j in range(len(scenarios))],
                       [np.mean(compute_latency[scenario][i]) \
                        for scenario in scenarios], color="k", linewidth=2,
                       marker="*", mfc="tab:pink", mew=1, mec="k", ms=25,
                       zorder=4.5)

    for scenario in scenarios:
        compute_means.append([np.mean(compute_latency[scenario][i]) \
                              for i in range(len(compute_latency[scenario]))])

    print(stage)
    baseline_index = 1
    for i, scenario in enumerate(scenarios):
        print(scenario)
        for j, batch_size in enumerate(common.batch_sizes[model]):
            print(f"  Load weight ({batch_size}):",
                  load_means[i][j] / load_means[baseline_index][j])
            print(f"  Compute ({batch_size}):",
                  compute_means[i][j] / compute_means[baseline_index][j])

    # format x-axis
    axis.set_xlabel("Batch Size", size=30)
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in common.batch_sizes[model]],
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

    # save the plot
    plt.savefig(common.plot_dir + \
                f"{model}_per_token_breakdown_{stage}_compressed.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

scenarios = ["NVDRAM", "MemoryMode", "NVDRAM (c)", "MemoryMode (c)", "DRAM (c)"]
scenario_dir = {
    "NVDRAM": "",
    "MemoryMode": "",
    "NVDRAM (c)": "compressed/",
    "MemoryMode (c)": "compressed/",
    "DRAM (c)": "compressed/"
}
colormap = {
    "NVDRAM": common.colormap[2],
    "MemoryMode": common.colormap[1],
    "NVDRAM (c)": common.colormap[0],
    "MemoryMode (c)": common.colormap[7],
    "DRAM (c)": common.colormap[8]
}
nvdram_config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
dram_config = "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
mm_config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"

load_weight_latency = {}
prefill_compute_latency = {}
decode_compute_latency = {}

warmup_completed = False

for scenario in scenarios:
    if "NVDRAM" in scenario:
        config = nvdram_config
    elif "MemoryMode" in scenario:
        config = mm_config
    elif "DRAM" in scenario:
        config = dram_config

    load_weight_latency[scenario] = []
    prefill_compute_latency[scenario] = []
    decode_compute_latency[scenario] = []

    for batch_size in common.batch_sizes[model]:
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

            elif line.startswith("load_weight (layer"):
                if warmup_completed:
                    load_weight_latency[scenario][-1].append(
                        float(line.split(" ")[-2]) * 1000)

            elif line.startswith("compute_layer_prefill  (per-batch)"):
                if warmup_completed:
                    prefill_compute_latency[scenario][-1].append(
                        float(line.split(" ")[-2]) * 1000)

            elif line.startswith("compute_layer_decoding (per-batch)"):
                if warmup_completed:
                    decode_compute_latency[scenario][-1].append(
                        float(line.split(" ")[-2]) * 1000)

gen_plot(load_weight_latency, prefill_compute_latency,
         scenarios, model, "prefill")
gen_plot(load_weight_latency, decode_compute_latency,
         scenarios, model, "decode")