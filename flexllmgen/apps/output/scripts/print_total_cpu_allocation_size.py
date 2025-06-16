#!/usr/bin/python3
import sys

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <allocation trace file>")
    sys.exit(-1)

alloc_trace_file = sys.argv[1]
alloc_size = 0

with open(alloc_trace_file, "r") as f:
    for line in f:
        if "Allocating weight on CPU" in line:
            alloc_size += int(line.split()[-1])

print("Total CPU allocation size: " + str(alloc_size / (2**30)) + " GB")