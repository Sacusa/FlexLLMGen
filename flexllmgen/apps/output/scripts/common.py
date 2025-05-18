from enum import Enum, auto
import numpy as np

################
# Applications #
################

class LLMModels(Enum):
    OPT_6B = auto()
    OPT_30B = auto()
    OPT_175B = auto()

app_name = {
        LLMModels.OPT_6B: "OPT-6.7B",
        LLMModels.OPT_30B: "OPT-30B",
        LLMModels.OPT_175B: "OPT-175B"
}

app_filename = {
        LLMModels.OPT_6B: "opt_6.7b.txt",
        LLMModels.OPT_30B: "opt_30b.txt",
        LLMModels.OPT_175B: "opt_175b.txt"
}

base_dir = "/u/sgupta45/FlexLLMGen/flexllmgen/apps/"
output_dir = base_dir + "output/"
plot_dir = output_dir + "plots/"

#####################
# Averaging methods #
#####################

amean = lambda i : sum(i) / len(i)

def gmean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))

##################
# Helper methods #
##################

bytes_to_mb = lambda x: x / (2 ** 20)
bytes_to_gb = lambda x: x / (2 ** 30)

######################
# Load/store methods #
######################

def get_llm_exec_time(model):
    exec_time = {}  # Dict with mem type as key (e.g., NVM)
    mem_type = ""
    data_distribution = ""

    for line in open(output_dir + app_filename[model]):
        if "Model generation time" in line:
            exec_time[mem_type][data_distribution].append(
                    int(line.split(" ")[4]))
        else:
            tokens = line.split(",")
            mem_type = tokens[0].strip()
            data_distribution = tokens[1].strip()

            if mem_type not in exec_time:
                exec_time[mem_type] = {}
            if data_distribution not in exec_time[mem_type]:
                exec_time[mem_type][data_distribution] = []

    return exec_time
