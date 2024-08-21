#!/bin/bash

# Function to execute the Python script with provided arguments in parallel
run_code() {
	python /home/sarah/Documentos/PSGCode/geexhp/parallel/genparallel_pc.py "$1" "$2"
}

# Export the function for use in parallel execution
export -f run_code

arguments=('0 8' '8 16' '16 24' '24 32' '32 40' '40 48' '48 56' '56 64' '64 72' '72 80' '80 88' '88 100')
for args in "${arguments[@]}"; do
  read start end <<< "$args"
  run_code "$start" "$end" &
done
