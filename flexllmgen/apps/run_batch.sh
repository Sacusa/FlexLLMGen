#!/bin/bash

NUM_PROMPTS=1
NUM_REPEATS=10
NUM_INSTANCES=1
BATCH_SIZE=1

SSD_OFFLOAD_DIR=/localtmp/sgupta45/flexgen/offload_dir
PMM_OFFLOAD_DIR=/mnt/pmem3/sgupta45/flexgen/offload_dir

OUTFILE_BASE=opt_30b_mm
BIN=./completion.py
# BIN="nsys profile \
#    --gpu-metrics-devices=0 \
#    --gpu-metrics-set=ga100 \
#    --output=${OUTFILE_BASE}_gpu_metrics_db \
#    --force-overwrite true \
#    ./completion.py"

# echo "opt-6.7b,ssd/na,nvm/na,dram/memory,gpu/dma"
# ${BIN} \
#     --model facebook/opt-6.7b \
#     --percent 100 0 100 0 100 0 \
#     --num-prompts ${NUM_PROMPTS} \
#     --num-generate-repeats ${NUM_REPEATS} \
#     --numa-nodes 0,1

# echo "opt-6.7b,ssd/storage,nvm/na,dram/memory,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#    echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-6.7b \
#         --percent 0 0 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 0,1 \
#         --debug-mode breakdown \
#         --print-debug-messages
# done

# echo "opt-6.7b,ssd/na,nvm/memory,dram/na,gpu/dma"
# ${BIN} \
#     --model facebook/opt-6.7b \
#     --percent 100 0 100 0 100 0 \
#     --num-prompts ${NUM_PROMPTS} \
#     --num-generate-repeats ${NUM_REPEATS} \
#     --numa-nodes 2,3

# echo "opt-30b,ssd/na,nvm/na,dram/memory,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#    echo "Instance ${i}"
#    ${BIN} \
#        --model facebook/opt-30b \
#        --batch-size ${BATCH_SIZE} \
#        --percent 60 40 100 0 100 0 \
#        --num-prompts ${NUM_PROMPTS} \
#        --num-generate-repeats ${NUM_REPEATS} \
#        --compress-weight \
#        --numa-nodes 0,1 \
#        --print-debug-messages \
#        --output-seq-len 21
#        # --debug-mode breakdown
# done

# echo "opt-30b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-30b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 60 40 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --compress-weight \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 21
#         # --debug-mode breakdown
# done

# echo "opt-30b,ssd/na,nvm/na,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-30b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 100 0 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done





# BASELINE

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 1 \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 8 \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-30b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-30b \
#         --batch-size 1 \
#         --percent 60 40 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-30b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-30b \
#         --batch-size 32 \
#         --percent 60 40 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# # COMPRESSED

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 1 \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 8 \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# # All-CPU

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 1 \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 8 \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 44 \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 1 \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 8 \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 44 \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# OWP

# sed -i 's/        WeightSchedulingPolicy.BASELINE/        WeightSchedulingPolicy.MLP_FOCUSED/g' /u/sgupta45/.local/lib/python3.10/site-packages/flexgen/flex_opt.py

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 1 \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size 1 \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done





echo "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
    echo "Instance ${i}"
    ${BIN} \
        --model facebook/opt-175b \
        --batch-size 3 \
        --percent 0 100 100 0 100 0 \
        --num-prompts 1 \
        --num-generate-repeats 2 \
        --numa-nodes 0,1 \
        --compress-weight \
        --print-debug-messages \
        --input-seq-len 1024 \
        --output-seq-len 1024 \
        --debug-mode breakdown
done

echo "opt-175b,ssd/na,nvm/na,dram/memory,gpu/dma"
for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
    echo "Instance ${i}"
    ${BIN} \
        --model facebook/opt-175b \
        --batch-size 1 \
        --percent 20 80 100 0 100 0 \
        --num-prompts 1 \
        --num-generate-repeats 2 \
        --numa-nodes 0,1 \
        --compress-weight \
        --print-debug-messages \
        --input-seq-len 1024 \
        --output-seq-len 1024 \
        --debug-mode breakdown
done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma-O1"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma-O2"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma-O3"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 2,3 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 0,1 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/na,gpu/dma-O5"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 0 100 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 0,1 \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/memory,dram/cache,gpu/dma"
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --batch-size ${BATCH_SIZE} \
#         --percent 20 80 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --compress-weight \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done

# echo "opt-175b,ssd/na,nvm/storage,dram/memory,gpu/dma"
# mkdir -p ${PMM_OFFLOAD_DIR}
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --offload-dir ${PMM_OFFLOAD_DIR} \
#         --batch-size ${BATCH_SIZE} \
#         --percent 20 15 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 0,1 \
#         --output-seq-len 21 \
#         --print-debug-messages
#         # --debug-mode breakdown \
# done

# echo "opt-175b,ssd/na,nvm/storage,dram/memory,gpu/dma"
# mkdir -p ${PMM_OFFLOAD_DIR}
# for (( i=0 ; i<${NUM_INSTANCES} ; i++ )); do
#     echo "Instance ${i}"
#     ${BIN} \
#         --model facebook/opt-175b \
#         --offload-dir ${PMM_OFFLOAD_DIR} \
#         --batch-size ${BATCH_SIZE} \
#         --percent 20 15 100 0 100 0 \
#         --num-prompts ${NUM_PROMPTS} \
#         --num-generate-repeats ${NUM_REPEATS} \
#         --numa-nodes 0,1 \
#         --print-debug-messages \
#         --output-seq-len 2 \
#         --debug-mode breakdown
# done
