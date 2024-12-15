#!/bin/bash

# Check for pv installation
if ! command -v pv &> /dev/null; then
    echo "Error: 'pv' is not installed. Please install it with:"
    echo "      sudo apt-get install pv"
    exit 1
fi

# Modes to process
modes=('modern' 'proterozoic' 'archean')

# Number of planets and threads
NPLANETS=20

# Number of threads (number of CPU cores)
NUM_THREADS=$(nproc)
if [ -z "$NUM_THREADS" ] || [ "$NUM_THREADS" -eq 0 ]; then
    NUM_THREADS=1
fi
echo ">> Running with $NUM_THREADS threads."

# Calculate planets per thread
planets_per_thread=$((NPLANETS / NUM_THREADS))

# Progress bar function
progress_bar() {
    local completed=$1
    local total=$2
    local width=50
    local progress=$((completed * width / total))
    local remaining=$((width - progress))
    printf "\r["
    printf "%0.s#" $(seq 1 $progress)
    printf "%0.s-" $(seq 1 $remaining)
    printf "] %d%%" $((completed * 100 / total))
}

# Initialize counters
total_skipped=0
total_done=0

# Loop over modes and execute Python script
for mode in "${modes[@]}"; do
    echo ""
    echo ">> Processing mode: $mode"
    args=()
    completed_tasks=0
    total_tasks=$NUM_THREADS

    # Temporary log file for this mode
    log_file=$(mktemp)
    pids=()

    # Track each thread's process
    for (( i=0; i<NUM_THREADS; i++ )); do
        start=$((planets_per_thread * i))
        if [ $i -eq $((NUM_THREADS - 1)) ]; then
            end=$NPLANETS
        else
            end=$((planets_per_thread * (i + 1)))
        fi
        args+=("$start" "$end")
        
        # Run the Python script in the background, redirecting output
        python genparallel_pc.py "$mode" "$start" "$end" >> "$log_file" 2>&1 &
        pids+=($!)
    done

    # Monitor progress
    while :; do
        completed_tasks=0
        for pid in "${pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                ((completed_tasks++))
            fi
        done
        progress_bar "$completed_tasks" "$total_tasks"
        if [ "$completed_tasks" -eq "$total_tasks" ]; then
            break
        fi
        sleep 0.1  # Short delay to reduce CPU usage while checking
    done
    echo ""

    # Parse the log file for errors and completions
    skipped=$(grep -c "Skipping..." "$log_file")
    successful=$((NPLANETS - skipped))
    echo ">> Mode '$mode': $successful planets processed, $skipped planets skipped."
    
    # Accumulate totals
    total_skipped=$((total_skipped + skipped))
    total_done=$((total_done + successful))

    # Clean up log file
    rm "$log_file"
done

# Final summary
echo ""
echo ">> Processing completed for all modes."
echo ">> Total planets processed: $total_done"
echo ">> Total planets skipped: $total_skipped"

