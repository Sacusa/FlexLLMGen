#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 18]
}
yticks = {
    "opt-175b": 3
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
                color=common.colormap_cxl[plot_number],
                hatch="//" if plot_number % 2 == 1 else "", zorder=3.5)
        # for i in x:
        #     plt.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                  elinewidth=4, markerfacecolor="k",
        #                  markeredgecolor="k", zorder=3.5)
        offset += width

    for i, scenario in enumerate(scenarios):
        if i % 2 == 0: continue
        print(scenario)
        for j, metric in enumerate(["TTFT", "TBT"]):
            print("  " + metric,
                  ((all_means[i-1][j] - all_means[i][j]) / \
                    all_means[i-1][j]) * 100)

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
    plt.savefig(common.plot_dir + f"{model}_tbt_ttft_owp_cxl.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

scenarios = ["O0", "O0_owp", "O3", "O3_owp", "O4", "O4_owp"]
scenario_dir = {
    "O0": "compressed/batch_size_1/",
    "O0_owp": "compressed/mlp_focused/batch_size_1/",
}
config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"

latency = {}

for scenario in scenarios:
    if "O3" in scenario or "O4" in scenario:
        # Compute TTFT and TBT for CXL
        ffn_load_latency, mha_load_latency = \
            common.get_cxl_ffn_mha_load_latency(scenario, model_size)
        ffn_compute_latency, mha_compute_latency = \
            common.get_cxl_ffn_mha_compute_latency(scenario, model_size, 1)

        ttft = (max(mha_compute_latency[0], ffn_load_latency) + \
                max(ffn_compute_latency[0], mha_load_latency)) * \
               ((common.num_layers[model] - 2) / 2)
        tbt = (max(mha_compute_latency[1], ffn_load_latency) + \
                max(ffn_compute_latency[1], mha_load_latency)) * \
              ((common.num_layers[model] - 2) / 2)

        latency[scenario] = [[ttft], [tbt]]
        continue

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