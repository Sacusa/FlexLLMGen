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

model_config_labels = {
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
    "O1_owp": "HeLM NVDRAM (c)",
    "O2_owp": "HeLM MemoryMode (c)",
    "O3_owp": "HeLM DRAM (c)",
    "O1_ac" : "All-CPU NVDRAM (c)",
    "O2_ac" : "All-CPU MemoryMode (c)",
    "O3_ac" : "All-CPU DRAM (c)"
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