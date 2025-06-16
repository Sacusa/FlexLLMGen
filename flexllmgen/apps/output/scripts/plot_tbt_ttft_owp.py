#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 6]
}
yticks = {
    "opt-175b": 1
}

def gen_plot(latency, scenarios, model):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = [0, 1]
    width, offset = common.get_width_offset(0.8, len(scenarios))

    all_means = []

    for plot_number, scenario in enumerate(scenarios):
        mean = []
        std = []
        for i in range(len(latency[scenario])):
            mean.append(np.mean(latency[scenario][i]))
            std.append(np.std(latency[scenario][i]))

        all_means.append(mean)

        plt.bar([i+offset for i in x], mean, edgecolor="black",
                label=common.scenario_labels[scenario], width=width,
                color=common.colormap[plot_number], zorder=3.5)
        # for i in x:
        #     plt.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                  elinewidth=4, markerfacecolor="k", markeredgecolor="k",
        #                  zorder=3.5)
        offset += width

    baseline_index = 3
    for i, scenario in enumerate(scenarios):
        print(scenario)
        for j, metric in enumerate(["TTFT", "TBT"]):
            print("  " + metric,
                  ((all_means[baseline_index][j] - all_means[i][j]) / \
                    all_means[baseline_index][j]) * 100)

    # format x-axis
    axis.set_xlabel("Metric", size=common.font_size["axis_label"])
    axis.set_xticks(x)
    axis.set_xticklabels(["TTFT", "TBT"],
                         size=common.font_size["axis_tick"])

    # format y-axis
    axis.set_ylabel("Latency (s)", size=common.font_size["axis_label"])
    axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    axis.set_ylim(ylim[model])
    axis.yaxis.set_major_locator(plt.MultipleLocator(yticks[model]))

    # format the plot
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=1, mode="expand", borderaxespad=0,
               fontsize=common.font_size["legend"])
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)

    # save the plot
    plt.savefig(common.plot_dir + f"{model}_tbt_ttft_owp.pdf",
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

    latency[scenario] = [[], []]  # 0: ttft, 1: tbt

    for line in open(common.output_dir + scenario_dir[scenario] + \
            f"opt_{model_size}_per_token_latency.txt"):
        line = line.strip()

        if line.startswith(model):
            if config_found:
                break
            elif line.startswith(config):
                config_found = True

        elif line.startswith("Throughput (0, 0)") and config_found:
            warmup_completed = True

        elif line.startswith("generate") and warmup_completed:
            l = float(line.split(" ")[-2])
            if "token 0" in line:
                latency[scenario][0].append(l)
            else:
                latency[scenario][1].append(l)

gen_plot(latency, scenarios, model)