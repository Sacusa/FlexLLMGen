#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-30b": [0, 45],
    "opt-175b": [0, 0.9]
}
yticks = {
    "opt-30b": 10,
    "opt-175b": 0.2
}

def gen_plot(throughput, configs, model):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(common.batch_sizes[model])))
    width, offset = common.get_width_offset(0.8, len(configs))

    all_means = []
    baseline_index = 2

    for plot_number, config in enumerate(configs):
        mean = []
        std = []
        for i in x:
            mean.append(np.mean(throughput[config][i]))
            std.append(np.std(throughput[config][i]))

        all_means.append(mean)

        plt.bar([i+offset for i in x], mean, edgecolor="black",
                label=common.model_config_labels[config], width=width,
                color=common.colormap[plot_number], zorder=3.5)
        # for i in x:
        #     plt.errorbar([i+offset], mean[i], std[i], ecolor="k",
        #                  elinewidth=4, markerfacecolor="k", markeredgecolor="k",
        #                  zorder=3.5)
        offset += width

    for config_index, config in enumerate(configs):
        print("  [Batch size impact] " + common.model_config_labels[config],
              (all_means[config_index][1] / \
               all_means[config_index][0]))
        print("  [Memory impact] " + common.model_config_labels[config] + "/" + \
              common.model_config_labels[configs[baseline_index]])
        for i in range(len(mean)):
            print(f"    BS={common.batch_sizes[model][i]}",
                  ((all_means[config_index][i] / \
                    all_means[baseline_index][i]) - 1) * 100)

    # format x-axis
    axis.set_xlabel("Batch Size", size=common.font_size["axis_label"])
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in common.batch_sizes[model]],
                         size=common.font_size["axis_tick"])

    # format y-axis
    axis.set_ylabel("Throughput (tokens/s)",
                    size=common.font_size["axis_label"])
    axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    axis.set_ylim(ylim[model])
    axis.yaxis.set_major_locator(plt.MultipleLocator(yticks[model]))

    # format the plot
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               ncol=2, mode="expand", borderaxespad=0,
               fontsize=common.font_size["legend"])
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)

    # save the plot
    plt.savefig(common.plot_dir + f"{model}_throughput.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

throughput = {}
configs = []

config = None
warmup_completed = False

for batch_size in common.batch_sizes[model]:
    for line in open(common.output_dir + \
            f"batch_size_{batch_size}/opt_{model_size}_per_token_latency.txt"):
        line = line.strip()

        if line.startswith(model):
            config = line.strip()
            if config not in configs:
                throughput[config] = [[]]
                configs.append(config)
            else:
                throughput[config].append([])
            warmup_completed = False

        elif line.startswith("Throughput"):
            if not line.startswith("Throughput (0, 0)"):
                tokens = line.split(" ")[-2]
                tokens = [float(t) for t in tokens.split("/")]
                throughput[config][-1].append(float(tokens[0] / \
                                                    common.ns_to_s(tokens[1])))

gen_plot(throughput, configs, model)