#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

batch_sizes = {
    "opt-175b": [8, 44]
}
ylim = {
    "opt-175b": [0, 14]
}
yticks = {
    "opt-175b": 2
}

def gen_plot(throughput, scenarios, model):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(batch_sizes[model])))
    width, offset = common.get_width_offset(0.8, len(scenarios))

    all_means = []

    for plot_number, scenario in enumerate(scenarios):
        mean = []
        std = []
        for i in x:
            if sum(throughput[scenario][i]) == 0:
                # skip empty batches
                mean.append(0)
                std.append(0)
            else:
                mean.append(np.mean(throughput[scenario][i]))
                std.append(np.std(throughput[scenario][i]))

        all_means.append(mean)

        plt.bar([i+offset for i in x], mean, edgecolor="black",
                label=common.scenario_labels[scenario], width=width,
                color=common.colormap_cxl[plot_number],
                hatch="//" if plot_number % 2 == 1 else "", zorder=3.5)
        # for i in x:
        #     plt.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                  elinewidth=4, markerfacecolor="k", markeredgecolor="k",
        #                  zorder=3.5)
        offset += width

    for i, scenario in enumerate(scenarios):
        if i % 2 == 0: continue
        print(scenario)
        print(" ", batch_sizes[model][0],
                ((all_means[i-1][0] - all_means[i][0]) / \
                all_means[i-1][0]) * 100)
        print(f"  {batch_sizes[model][1]}/{batch_sizes[model][0]}",
                all_means[i][1] / all_means[i][0])

    # format x-axis
    axis.set_xlabel("Batch Size", size=common.font_size["axis_label"])
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in batch_sizes[model]],
                         size=common.font_size["axis_tick"])

    # format y-axis
    axis.set_ylabel("Throughput (tokens/s)",
                    size=common.font_size["axis_label"])
    axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    axis.set_ylim(ylim[model])
    axis.yaxis.set_major_locator(plt.MultipleLocator(yticks[model]))

    # format the plot
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=1, mode="expand", borderaxespad=0,
               fontsize=common.font_size["legend"])
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)

    # save the plot
    plt.savefig(common.plot_dir + f"{model}_throughput_ac_cxl.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

scenarios = ["O0", "O0_ac", "O3", "O3_ac", "O4", "O4_ac"]
scenario_dir = {
    "O0"   : "compressed/",
    "O0_ac": "compressed/all_cpu/",
}
config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"

throughput = {}

for scenario in scenarios:
    throughput[scenario] = []

    for batch_size in batch_sizes[model]:
        if "ac" not in scenario and batch_size > 8:
            # These configs only supports batch size up to 8
            throughput[scenario].append([0])
            continue

        elif "O3" in scenario or "O4" in scenario:
            # Compute throughput for CXL
            ffn_load_latency, mha_load_latency = \
                common.get_cxl_ffn_mha_load_latency(scenario, model_size)
            ffn_compute_latency, mha_compute_latency = \
                common.get_cxl_ffn_mha_compute_latency(scenario, model_size,
                                                       batch_size)
            num_tokens = 21
            multiplier = {1: 1.58, 8: 1.57, 44: 1.45}

            ttft = (max(mha_compute_latency[0], ffn_load_latency) + \
                    max(ffn_compute_latency[0], mha_load_latency)) * \
                   ((common.num_layers[model] - 2) / 2)
            tbt = (max(mha_compute_latency[1], ffn_load_latency) + \
                   max(ffn_compute_latency[1], mha_load_latency)) * \
                  ((common.num_layers[model] - 2) / 2)

            throughput[scenario].append(
                [(num_tokens * batch_size * multiplier[batch_size]) / \
                 (ttft + (tbt * (num_tokens - 1)))])
            # throughput[scenario].append([batch_size / tbt])
            continue

        throughput[scenario].append([])

        config_found = False

        for line in open(common.output_dir + scenario_dir[scenario] + \
                (f"batch_size_{batch_size}/opt_{model_size}"
                 "_per_token_latency.txt")):
            line = line.strip()

            if line.startswith(model):
                if config_found:
                    break
                elif line.startswith(config):
                    config_found = True

            elif line.startswith("Throughput"):
                if not line.startswith("Throughput (0, 0)"):
                    tokens = line.split(" ")[-2]
                    tokens = [float(t) for t in tokens.split("/")]
                    throughput[scenario][-1].append(
                        float(tokens[0] / common.ns_to_s(tokens[1])))

gen_plot(throughput, scenarios, model)