#!/usr/bin/python3
import common
import csv
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <model size>")
    sys.exit(1)

model_size = sys.argv[1]
model = "opt-" + model_size
last_layer = common.num_layers[model] - 1

base_scenarios = ["O0", "O1", "O3", "O4"]
base_print_scenarios = ["O0", "O3", "O4"]
scenario_dir = {
    "O0"   : "compressed/",
    "O1"   : "compressed/",
    "O0_owp": "compressed/mlp_focused/",
    "O1_owp": "compressed/mlp_focused/",
    "O0_ac": "compressed/all_cpu/",
    "O1_ac": "compressed/all_cpu/",
}
suffixes = ["", "_owp", "_ac"]
suffix_labels = {
    "": "Baseline",
    "_owp": "HeLM",
    "_ac": "All-CPU"
}
batch_sizes = {
    "": common.batch_sizes[model],
    "_owp": [1],
    "_ac": common.all_cpu_batch_sizes[model]
}
nvdram_config = "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
dram_config = "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
mm_config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"

with open(common.plot_dir + \
          f"{model}_compute_load_overlap_cxl_all.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header rows
    data = [["Allocation", "Batch", "Stage"] + \
            ["MHA compute/FFN Load (ratio)"] + \
            [""] * (len(base_print_scenarios) - 1) + \
            ["FFN Compute/MHA Load (ratio)"] + \
            [""] * (len(base_print_scenarios) - 1),

            ["Policy", "Size", ""] + ([common.scenario_labels[s] \
                         for s in base_print_scenarios]) * 2
           ]
    csvwriter.writerows(data)

    for suffix in suffixes:
        scenarios = [bs + suffix for bs in base_scenarios]
        print_scenarios = [bs + suffix for bs in base_print_scenarios]

        for batch_size in batch_sizes[suffix]:
            prefill_load_latency = {}
            prefill_compute_latency = {}
            decode_load_latency = {}
            decode_compute_latency = {}

            warmup_completed = False
            in_prefill = True

            for scenario in scenarios:
                if "O1" in scenario:
                    config = mm_config
                elif "O3" in scenario or "O4" in scenario:
                    continue
                else:
                    config = nvdram_config

                config_found = False
                warmup_completed = False
                in_prefill = True

                prefill_load_latency[scenario] = []
                prefill_compute_latency[scenario] = []
                decode_load_latency[scenario] = []
                decode_compute_latency[scenario] = []

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

                    elif line.startswith("load_weight (layer ") and \
                        warmup_completed:
                        if "layer 0" in line or f"layer {last_layer}" in line:
                            # NOTE: SPECIAL HANDLING FOR NVDRAM
                            ###################################
                            if "O0" in scenario and \
                               f"layer {last_layer}" in line:
                                in_prefill = False
                            continue

                        if in_prefill:
                            prefill_load_latency[scenario].append(
                                float(line.split(" ")[-2]) * 1000)
                        else:
                            decode_load_latency[scenario].append(
                                float(line.split(" ")[-2]) * 1000)

                    elif line.startswith("compute_layer (layer ") and \
                        warmup_completed:
                        if "layer 0" in line:
                            continue
                        elif f"layer {last_layer}" in line:
                            in_prefill = False
                            continue

                        if in_prefill:
                            prefill_compute_latency[scenario].append(
                                float(line.split(" ")[-2]) * 1000)
                        else:
                            decode_compute_latency[scenario].append(
                                float(line.split(" ")[-2]) * 1000)

            prefill_mha_load_latency = {}
            prefill_ffn_load_latency = {}
            prefill_mha_compute_latency = {}
            prefill_ffn_compute_latency = {}

            decode_mha_load_latency = {}
            decode_ffn_load_latency = {}
            decode_mha_compute_latency = {}
            decode_ffn_compute_latency = {}

            for scenario in scenarios:
                if "O3" in scenario or "O4" in scenario:
                    ffn_load_latency, mha_load_latency = \
                        common.get_cxl_ffn_mha_load_latency(scenario,
                                                            model_size)
                    ffn_load_latency *= 1000
                    mha_load_latency *= 1000
                    prefill_mha_load_latency[scenario] = mha_load_latency
                    prefill_ffn_load_latency[scenario] = ffn_load_latency
                    decode_mha_load_latency[scenario] = mha_load_latency
                    decode_ffn_load_latency[scenario] = ffn_load_latency
                else:
                    prefill_mha_load_latency[scenario] = \
                        prefill_load_latency[scenario][0::2]
                    prefill_ffn_load_latency[scenario] = \
                        prefill_load_latency[scenario][1::2]

                    decode_mha_load_latency[scenario] = \
                        decode_load_latency[scenario][0::2]
                    decode_ffn_load_latency[scenario] = \
                        decode_load_latency[scenario][1::2]

                if "O1" in scenario:
                    prefill_mha_compute_latency[scenario] = \
                        prefill_compute_latency[scenario][0::2]
                    prefill_ffn_compute_latency[scenario] = \
                        prefill_compute_latency[scenario][1::2]
                    decode_mha_compute_latency[scenario] = \
                        decode_compute_latency[scenario][0::2]
                    decode_ffn_compute_latency[scenario] = \
                        decode_compute_latency[scenario][1::2]

            # # NOTE: SPECIAL HANDLING FOR NVDRAM and CXL
            # ###########################################
            # copy_scenario = scenarios[base_scenarios.index("O1")]
            # for scenario in print_scenarios:
            #     prefill_mha_compute_latency[scenario] = \
            #         prefill_mha_compute_latency[copy_scenario][:]
            #     prefill_ffn_compute_latency[scenario] = \
            #         prefill_ffn_compute_latency[copy_scenario][:]
            #     decode_mha_compute_latency[scenario] = \
            #         decode_mha_compute_latency[copy_scenario][:]
            #     decode_ffn_compute_latency[scenario] = \
            #         decode_ffn_compute_latency[copy_scenario][:]

            compute_scenario = scenarios[base_scenarios.index("O1")]

            # Write prefill data
            data = [suffix_labels[suffix], str(batch_size), "Prefill"]
            data += ["{:0.2f}".format(np.mean(
                prefill_mha_compute_latency[compute_scenario]) / \
                np.mean(prefill_ffn_load_latency[s])) for s in print_scenarios]
            data += ["{:0.2f}".format(
                np.mean(prefill_ffn_compute_latency[compute_scenario]) / \
                np.mean(prefill_mha_load_latency[s])) for s in print_scenarios]
            csvwriter.writerow(data)

            # Write decode data
            data = [suffix_labels[suffix], str(batch_size), "Decode"]
            data += ["{:0.2f}".format(
                np.mean(decode_mha_compute_latency[compute_scenario]) / \
                np.mean(decode_ffn_load_latency[s])) for s in print_scenarios]
            data += ["{:0.2f}".format(
                np.mean(decode_ffn_compute_latency[compute_scenario]) / \
                np.mean(decode_mha_load_latency[s])) for s in print_scenarios]
            csvwriter.writerow(data)