# FlexLLMGen: High-throughput Generative Inference of Large Language Models with a Single GPU [[paper](https://arxiv.org/abs/2303.06865)]

FlexLLMGen is a high-throughput generation engine for running large language models with limited GPU memory. FlexLLMGen allows **high-throughput** generation by IO-efficient offloading, compression, and **large effective batch sizes**.

----------

## Installation
Requirements:
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

```
git clone https://github.com/FMInference/FlexLLMGen.git
cd FlexLLMGen
pip install -e .
```

## Usage and Examples

To get started, you can try a small model like OPT-1.3B first. It fits into a single GPU so no offloading is required.
FlexLLMGen will automatically download weights from Hugging Face. Navigate to `flexllmgen/apps` and run:

```
python3 completion.py --model facebook/opt-1.3b
```