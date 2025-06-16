#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 15]
}
yticks = {
    "opt-175b": 3
}

def gen_plot(latency, scenarios, model, metric):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = list(range(len(common.all_cpu_batch_sizes[model])))
    width, offset = common.get_width_offset(0.8, len(scenarios))

    all_means = []

    for plot_number, scenario in enumerate(scenarios):
        mean = []
        std = []
        for i in range(len(x)):
            if sum(latency[scenario][i]) == 0:
                # skip empty batches
                mean.append(0)
                std.append(0)
            else:
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

    baseline_index = [1, 1, 1]
    print(metric)
    for i, scenario in enumerate(scenarios):
        print(scenario)
        for j, bs in enumerate(common.all_cpu_batch_sizes[model]):
            print(" ", bs,
                  ((all_means[baseline_index[j]][j] - all_means[i][j]) / \
                    all_means[baseline_index[j]][j]) * 100)

    # format x-axis
    axis.set_xlabel("Batch Size", size=common.font_size["axis_label"])
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in common.all_cpu_batch_sizes[model]],
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
    plt.savefig(common.plot_dir + f"{model}_{metric}_ac.pdf",
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

ttft = {}
tbt = {}

for scenario in scenarios:
    if scenario == "O2_ac":
        config = mm_config
    elif scenario == "O3_ac":
        config = dram_config
    else:
        config = nvdram_config

    ttft[scenario] = []
    tbt[scenario] = []

    for batch_size in common.all_cpu_batch_sizes[model]:
        if scenario == "O0" and batch_size > 8:
            # DRAM config only supports batch size 1
            ttft[scenario].append([0])
            tbt[scenario].append([0])
            continue

        ttft[scenario].append([])
        tbt[scenario].append([])

        config_found = False
        warmup_completed = False

        for line in open(common.output_dir + scenario_dir[scenario] + \
                (f"batch_size_{batch_size}/opt_{model_size}"
                 "_per_token_latency.txt")):
            line = line.strip()

            if line.startswith(model):
                if config_found:
                    break
                elif line.startswith(config):
                    config_found = True

            elif line.startswith("Throughput (0, 0)") and config_found:
                warmup_completed = True

            elif line.startswith("generate") and warmup_completed:
                latency = float(line.split(" ")[-2])
                if "token 0" in line:
                    ttft[scenario][-1].append(latency)
                else:
                    tbt[scenario][-1].append(latency)

gen_plot(tbt,  scenarios, model, "tbt")
gen_plot(ttft, scenarios, model, "ttft")