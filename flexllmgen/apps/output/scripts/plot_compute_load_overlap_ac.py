#!/usr/bin/python3
import common
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

ylim = {
    "opt-175b": [0, 100]
}
yticks = {
    "opt-175b": 20
}

scenarios = ["O0", "O0_ac", "O1_ac", "O2_ac"]
scenario_dir = {
    "O0"   : "compressed/",
    "O1"   : "compressed/",
    "O0_ac": "compressed/all_cpu/",
    "O1_ac": "compressed/all_cpu/",
    "O2_ac": "compressed/all_cpu/"
}
nvdram_scenarios = ["O0", "O0_ac"]
batch_size = {"O0": 8, "O0_ac": 44, "O1_ac": 44, "O2_ac": 44}

nvdram_config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
dram_config = "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
mm_config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"

def get_stats(model, scenario, batch_size):
    last_layer = common.num_layers[model] - 1

    if scenario == "O1_ac":
        config = mm_config
    elif scenario == "O2_ac":
        config = dram_config
    else:
        config = nvdram_config

    config_found = False
    warmup_completed = False
    in_prefill = True

    prefill_load_latency = []
    prefill_compute_latency = []
    decode_load_latency = []
    decode_compute_latency = []

    for line in open(common.output_dir + scenario_dir[scenario] + \
            (f"batch_size_{batch_size}/opt_{model_size}"
                "_exec_time_breakdown.txt")):
        line = line.strip()

        if line.startswith(model):
            if config_found:
                break
            elif line.startswith(config):
                config_found = True

        elif line.startswith("Throughput") and config_found:
            if line.startswith("Throughput (0, 0)"):
                warmup_completed = True
            in_prefill = True

        elif line.startswith("load_weight (layer ") and warmup_completed:
            if "layer 0" in line or f"layer {last_layer}" in line:
                # NOTE: SPECIAL HANDLING FOR NVDRAM
                ###################################
                if scenario in ["O0", "O0_ac"] and \
                    f"layer {last_layer}" in line:
                    in_prefill = False
                continue

            if in_prefill:
                prefill_load_latency.append(
                    float(line.split(" ")[-2]) * 1000)
            else:
                decode_load_latency.append(
                    float(line.split(" ")[-2]) * 1000)

        elif line.startswith("compute_layer (layer ") and warmup_completed:
            if "layer 0" in line:
                continue
            elif f"layer {last_layer}" in line:
                in_prefill = False
                continue

            if in_prefill:
                prefill_compute_latency.append(
                    float(line.split(" ")[-2]) * 1000)
            else:
                decode_compute_latency.append(
                    float(line.split(" ")[-2]) * 1000)

    prefill_mha_load_latency = prefill_load_latency[0::2]
    prefill_ffn_load_latency = prefill_load_latency[1::2]
    decode_mha_load_latency = decode_load_latency[0::2]
    decode_ffn_load_latency = decode_load_latency[1::2]

    # NOTE: SPECIAL HANDLING FOR NVDRAM
    ###################################
    if scenario not in nvdram_scenarios:
        prefill_mha_compute_latency = prefill_compute_latency[0::2]
        prefill_ffn_compute_latency = prefill_compute_latency[1::2]
        decode_mha_compute_latency = decode_compute_latency[0::2]
        decode_ffn_compute_latency = decode_compute_latency[1::2]
    else:
        prefill_mha_compute_latency = []
        prefill_ffn_compute_latency = []
        decode_mha_compute_latency = []
        decode_ffn_compute_latency = []

    return (prefill_mha_load_latency, prefill_ffn_load_latency,
            prefill_mha_compute_latency, prefill_ffn_compute_latency,
            decode_mha_load_latency, decode_ffn_load_latency,
            decode_mha_compute_latency, decode_ffn_compute_latency)

def gen_plot(mha_load_latency, mha_compute_latency, ffn_load_latency,
             ffn_compute_latency, scenarios, model, stage):
    # initialize pyplot
    plt.clf()
    _, axis = plt.subplots(figsize=common.figsize_small, dpi=600)
    plt.rc('axes', axisbelow=True)

    # plot parameters
    x = [0, 1]  # 0: MHA compute/FFN load, 1: FFN compute/MHA load
    width, offset = common.get_width_offset(0.8, len(scenarios))

    load_means = []

    for plot_number, scenario in enumerate(scenarios):
        mean = [np.mean(ffn_load_latency[scenario]),
                np.mean(mha_load_latency[scenario])]

        # bars for load_weight_latency
        axis.bar([i+offset for i in x], mean, edgecolor="black",
                 label=f"{common.scenario_labels[scenario]}"
                       f"({batch_size[scenario]})", width=width,
                 color=common.colormap[plot_number], zorder=3.5)
        offset += width

        load_means.append(mean)

    width, offset = common.get_width_offset(0.8, len(scenarios))
    for i, compute_latency in enumerate([mha_compute_latency,
                                         ffn_compute_latency]):
        axis.plot([i + offset + (width * j) for j in range(len(scenarios))],
                  [np.mean(compute_latency[scenario]) \
                   for scenario in scenarios], color="k", linewidth=2,
                  marker="*", mfc="tab:pink", mew=1, mec="k", ms=25, zorder=4.5)

    print(stage)
    baseline_index = 0
    for i, scenario in enumerate(scenarios):
        print(scenario)
        print("  FFN load:", load_means[i][0] / load_means[baseline_index][0])
        print("  MHA load:", load_means[i][1] / load_means[baseline_index][1])

    # format x-axis
    # axis.set_xlabel("Stage", size=30)
    axis.set_xticks(x)
    axis.set_xticklabels(["MHA Compute\nFFN Load", "FFN Compute\nMHA Load"],
                         size=common.font_size["axis_tick"])

    # format y-axis
    axis.set_ylabel("Time (ms)", size=common.font_size["axis_label"])
    axis.yaxis.set_tick_params(labelsize=common.font_size["axis_tick"])
    axis.set_ylim(ylim[model])
    axis.yaxis.set_major_locator(plt.MultipleLocator(yticks[model]))

    # format the plot
    axis.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                ncol=1, mode="expand", borderaxespad=0,
                fontsize=common.font_size["legend"])
    axis.grid(zorder=2.5, axis='y', color='silver', linestyle='-', linewidth=2)

    # save the plot
    plt.savefig(common.plot_dir + \
                f"{model}_compute_load_overlap_ac_{stage}.pdf",
        bbox_inches="tight")

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size

prefill_mha_load_latency = {}
prefill_ffn_load_latency = {}
prefill_mha_compute_latency = {}
prefill_ffn_compute_latency = {}
decode_mha_load_latency = {}
decode_ffn_load_latency = {}
decode_mha_compute_latency = {}
decode_ffn_compute_latency = {}

for scenario in scenarios:
    prefill_mha_load, prefill_ffn_load, prefill_mha_compute, \
        prefill_ffn_compute, decode_mha_load, decode_ffn_load, \
        decode_mha_compute, decode_ffn_compute = \
        get_stats(model, scenario, batch_size[scenario])

    # NOTE: SPECIAL HANDLING FOR NVDRAM
    ###################################
    if scenario in nvdram_scenarios:
        if scenario == "O0":
            _, _, prefill_mha_compute, prefill_ffn_compute, _, _, \
                decode_mha_compute, decode_ffn_compute = \
                get_stats(model, "O1", batch_size[scenario])
        elif scenario == "O0_ac":
            _, _, prefill_mha_compute, prefill_ffn_compute, _, _, \
                decode_mha_compute, decode_ffn_compute = \
                get_stats(model, "O1_ac", batch_size[scenario])

    prefill_mha_load_latency[scenario] = prefill_mha_load
    prefill_ffn_load_latency[scenario] = prefill_ffn_load
    prefill_mha_compute_latency[scenario] = prefill_mha_compute
    prefill_ffn_compute_latency[scenario] = prefill_ffn_compute
    decode_mha_load_latency[scenario] = decode_mha_load
    decode_ffn_load_latency[scenario] = decode_ffn_load
    decode_mha_compute_latency[scenario] = decode_mha_compute
    decode_ffn_compute_latency[scenario] = decode_ffn_compute

gen_plot(prefill_mha_load_latency, prefill_mha_compute_latency,
            prefill_ffn_load_latency, prefill_ffn_compute_latency,
            scenarios, model, "prefill")
gen_plot(decode_mha_load_latency, decode_mha_compute_latency,
            decode_ffn_load_latency, decode_ffn_compute_latency,
            scenarios, model, "decode")