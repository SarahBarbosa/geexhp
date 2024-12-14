#!/bin/bash

# Modes to process
modes=('modern' 'proterozoic' 'archean')

# Number of planets and threads
NPLANETS=50000

# Number of threads (number of CPU cores)
NUM_THREADS=$(nproc)
if [ -z "$NUM_THREADS" ] || [ "$NUM_THREADS" -eq 0 ]; then
    NUM_THREADS=1
fi
echo ">> Running with $NUM_THREADS threads."

# Calculate planets per thread
planets_per_thread=$((NPLANETS / NUM_THREADS))

# Loop over modes and execute Python script
for mode in "${modes[@]}"; do
    echo ">> Processing mode: $mode"
    args=()
    for (( i=0; i<NUM_THREADS; i++ )); do
        start=$((planets_per_thread * i))
        if [ $i -eq $((NUM_THREADS - 1)) ]; then
            end=$NPLANETS
        else
            end=$((planets_per_thread * (i + 1)))
        fi
        args+=("$start" "$end")
    done
    # Pass start and end as separate arguments
    python genparallel_pc.py "$mode" "${args[@]}"
done

