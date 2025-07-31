from enum import Enum, auto
import matplotlib
import numpy as np

base_dir = "/u/sgupta45/FlexLLMGen/flexllmgen/apps/"
output_dir = base_dir + "output/"
plot_dir = output_dir + "plots/"

#############################
# Model specific parameters #
#############################

num_layers = {
    "opt-30b": 98,
    "opt-175b": 194
}
batch_sizes = {
    "opt-30b": [1, 32],
    "opt-175b": [1, 8]
}
all_cpu_batch_sizes = {
    "opt-175b": [1, 8, 44]
}
host_layer_size_compressed = {  # in bytes
    "opt-175b": {
        "baseline": {
            "mha": 254877696,
            "ffn": 679575552
        },
        "owp": {
            "mha": 339738624,
            "ffn": 339738624
        },
        "ac": {
            "mha": 339886080,
            "ffn": 679649280
        }
    }
}

#####################
# Averaging methods #
#####################

amean = lambda i : sum(i) / len(i)

def gmean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))

#######################
# Plotting parameters #
#######################

colormap = matplotlib.cm.get_cmap("tab10").colors
colormap_cxl = matplotlib.cm.get_cmap("tab20").colors

model_config_labels = {
    "opt-30b,ssd/na,nvm/na,dram/na,gpu/dma": "GPU",
    "opt-30b,ssd/na,nvm/na,dram/memory,gpu/dma": "DRAM",
    "opt-30b,ssd/na,nvm/memory,dram/na,gpu/dma": "NVDRAM",
    "opt-30b,ssd/na,nvm/memory,dram/cache,gpu/dma": "MemoryMode",
    "opt-175b,ssd/storage,nvm/na,dram/memory,gpu/dma": "SSD",
    "opt-175b,ssd/na,nvm/storage,dram/memory,gpu/dma": "FSDAX",
    "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma": "NVDRAM",
    "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma": "MemoryMode",
    "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma": "DRAM",
}

figsize_small = (9, 6)
figsize_medium = (12, 6)
figsize_large = (24, 6)

font_size = {
    "axis_label": 35,
    "axis_tick": 30,
    "legend": 28,
}

breakdown_labels = {
    "load_weight": "Weight Transfer Time (ms)",
    "compute": "Compute Time (ms)",
}

scenario_labels = {
    "O0"    : "NVDRAM (c)",
    "O1"    : "MemoryMode (c)",
    "O2"    : "DRAM (c)",
    "O0_owp": "HeLM NVDRAM (c)",
    "O1_owp": "HeLM MemoryMode (c)",
    "O2_owp": "HeLM DRAM (c)",
    "O0_ac" : "All-CPU NVDRAM (c)",
    "O1_ac" : "All-CPU MemoryMode (c)",
    "O2_ac" : "All-CPU DRAM (c)",

    # CXL scenarios
    "O3"     : "CXL-FPGA (c)",
    "O4"     : "CXL-ASIC (c)",
    "O3_owp" : "HeLM CXL-FPGA (c)",
    "O4_owp" : "HeLM CXL-ASIC (c)",
    "O3_ac"  : "All-CPU CXL-FPGA (c)",
    "O4_ac"  : "All-CPU CXL-ASIC (c)",
}

##################
# Helper methods #
##################

bytes_to_mb = lambda x: x / (2 ** 20)
bytes_to_gb = lambda x: x / (2 ** 30)
ns_to_s = lambda x: x / 1e9

def get_width_offset(max_width, num_bars):
    width = max_width / num_bars
    if num_bars % 2 == 0:
        offset = -width * (0.5 + ((num_bars / 2) - 1))
    else:
        offset = -width * ((num_bars - 1) / 2)
    return width, offset

######################
# Projection methods #
######################

cxl_read_bandwidth = {  # in GB/s
    # "O3": 19.2,
    "O3": 5.12,
    "O4": 28
}

def get_cxl_ffn_mha_load_latency(scenario, model_size):
    base_scenario = scenario[:2]
    model = "opt-" + model_size

    if len(scenario) > 2:
        modifier = scenario[3:]
    else:
        modifier = "baseline"

    return (bytes_to_gb(host_layer_size_compressed[model][modifier]["ffn"]) / \
                cxl_read_bandwidth[base_scenario],
            bytes_to_gb(host_layer_size_compressed[model][modifier]["mha"]) / \
                cxl_read_bandwidth[base_scenario])

def get_cxl_ffn_mha_compute_latency(scenario, model_size, batch_size):
    if len(scenario) > 2:
        modifier = scenario[2:]
    else:
        modifier = ""

    modifier_dir = {
        ""    : "compressed/",
        "_owp": "compressed/mlp_focused/",
        "_ac" : "compressed/all_cpu/"
    }

    model = "opt-" + model_size
    last_layer = num_layers[model] - 1

    config = "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma" # using MM config

    config_found = False
    warmup_completed = False
    in_prefill = True

    prefill_compute_latency = []
    decode_compute_latency = []

    for line in open(output_dir + modifier_dir[modifier] + \
        (f"batch_size_{batch_size}/opt_{model_size}_exec_time_breakdown.txt")):
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

        elif line.startswith("compute_layer (layer ") and warmup_completed:
            if "layer 0" in line:
                continue
            elif f"layer {last_layer}" in line:
                in_prefill = False
                continue

            latency = float(line.split(" ")[-2])
            if in_prefill:
                prefill_compute_latency.append(latency)
            else:
                decode_compute_latency.append(latency)

    prefill_mha_compute_latency = np.mean(prefill_compute_latency[0::2])
    prefill_ffn_compute_latency = np.mean(prefill_compute_latency[1::2])
    decode_mha_compute_latency = np.mean(decode_compute_latency[0::2])
    decode_ffn_compute_latency = np.mean(decode_compute_latency[1::2])

    return ((prefill_ffn_compute_latency, decode_ffn_compute_latency),
            (prefill_mha_compute_latency, decode_mha_compute_latency))