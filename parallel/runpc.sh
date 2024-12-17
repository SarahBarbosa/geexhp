#!/bin/bash

# Colors
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'
RESET='\033[0m'

# Trap Ctrl+C to clean up
trap "echo -e '\n\nScript interrupted. Cleaning up...'; pkill -P $$; rm -f progress_*.tmp; exit" SIGINT

# Check for pv installation
if ! command -v pv &> /dev/null; then
    echo -e "${YELLOW}Error: 'pv' is not installed. Please install it with:${RESET}"
    echo "      sudo apt-get install pv"
    exit 1
fi

# Prompt user for the number of planets
echo -e "${CYAN}Enter the number of planets to process: ${RESET}\c"
read NPLANETS

# Validate user input
if ! [[ "$NPLANETS" =~ ^[0-9]+$ ]]; then
    echo -e "${YELLOW}Error: NPLANETS must be a positive integer.${RESET}"
    exit 1
fi

# Modes to process
modes=('modern' 'proterozoic' 'archean')

# Number of threads (number of CPU cores)
NUM_THREADS=$(nproc)
if [ -z "$NUM_THREADS" ] || [ "$NUM_THREADS" -eq 0 ]; then
    NUM_THREADS=1
fi

echo ""
echo -e ">> Running with $NUM_THREADS threads."

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

# Sum progress from all progress files
get_total_progress() {
    local total=0
    for file in progress_*.tmp; do
        [[ -f "$file" ]] && total=$((total + $(wc -l < "$file")))
    done
    echo "$total"
}

# Function to format time as hours, minutes, and seconds
format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

# Initialize counters
total_skipped=0
total_done=0
script_start_time=$(date +%s)

# Loop over modes and execute Python script
for mode in "${modes[@]}"; do
    echo ""
    echo -e "${CYAN}>> Processing mode: $mode${RESET}"
    log_file=$(mktemp)
    pids=()

    # Track start time for this mode
    mode_start_time=$(date +%s)

    # Launch processes
    for (( i=0; i<NUM_THREADS; i++ )); do
        start=$((planets_per_thread * i))
        end=$((i == NUM_THREADS-1 ? NPLANETS : start + planets_per_thread))
        python genparallel_pc.py "$mode" "$start" "$end" >> "$log_file" 2>&1 &
        pids+=($!)
    done

    # Monitor progress
    while :; do
        total_progress=$(get_total_progress)
        progress_bar "$total_progress" "$NPLANETS"
        [[ "$total_progress" -ge "$NPLANETS" ]] && break
        sleep 0.2
    done

    # Wait for all processes and check for errors
    for pid in "${pids[@]}"; do
        wait "$pid" || echo -e "${YELLOW}Warning: A thread exited with an error.${RESET}"
    done

    # Calculate mode duration
    mode_end_time=$(date +%s)
    mode_duration=$((mode_end_time - mode_start_time))

    # Parse skipped planets
    skipped=$(grep -c "Skipping..." "$log_file" || echo 0)
    successful=$((NPLANETS - skipped))

    # Display results for this mode
    echo -e "\nMode '${CYAN}$mode${RESET}':"
    printf "    ${GREEN}✅ %-5d planets processed${RESET}\n" "$successful"
    printf "    ${YELLOW}⚠️  %-5d planets skipped${RESET}\n" "$skipped"
    printf "    ${BLUE}⏱️  Duration: %s${RESET}\n" "$(format_time $mode_duration)"

    # Accumulate totals
    total_skipped=$((total_skipped + skipped))
    total_done=$((total_done + successful))

    # Clean up
    rm -f progress_*.tmp "$log_file"
done

# Final summary
script_end_time=$(date +%s)
total_duration=$((script_end_time - script_start_time))

echo ""
echo -e "${CYAN}-----------------------------------------${RESET}"
echo -e "${CYAN}   PROCESSING COMPLETED FOR ALL MODES${RESET}"
echo -e "${CYAN}-----------------------------------------${RESET}"
printf "    ${GREEN}✅ Total planets processed: %-5d${RESET}\n" "$total_done"
printf "    ${YELLOW}⚠️  Total planets skipped:   %-5d${RESET}\n" "$total_skipped"
printf "    ${BLUE}⏱️  Total Duration:          %s${RESET}\n" "$(format_time $total_duration)"
echo ""
echo -e "${YELLOW}Note:${RESET} The planets were skipped because:"
echo -e "         Exhausted all attempts to find a planet configuration that can retain a stable atmosphere with liquid water."

