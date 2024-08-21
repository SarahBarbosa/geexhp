import os
from typing import List, Tuple

NUM_THREADS = os.cpu_count()

def threadranges(nplanets: int, num_threads: int) -> List[Tuple[int, int]]:
    """
    Calculate the workload ranges for each thread.
    """
    planets_per_thread = nplanets // num_threads
    thread_ranges = []
    
    for i in range(num_threads):
        start = planets_per_thread * i
        end = planets_per_thread * (i + 1) if i != num_threads - 1 else nplanets
        thread_ranges.append((start, end))
    
    return thread_ranges

def parallel_bash_script(nplanets: int) -> None:
    """
    Generate a bash script to run the Python script in parallel using multiple threads.

    None
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(script_dir, f"N_{nplanets}_multithread_pc.sh")
    python_script_path = os.path.join(script_dir, "genparallel_pc.py")
    
    bash_script_content = [
        "#!/bin/bash\n\n",
        "# Function to execute the Python script with provided arguments in parallel\n",
        "run_code() {\n\t",
        f"python {python_script_path} \"$1\" \"$2\"\n",
        "}\n\n",
        "# Export the function for use in parallel execution\n",
        "export -f run_code\n\n"
    ]

    ranges = threadranges(nplanets, NUM_THREADS)
    formatted_arguments = ' '.join([f"'{start} {end}'" for start, end in ranges])

    bash_script_content += [
        f"arguments=({formatted_arguments})\n",
        "for args in \"${arguments[@]}\"; do\n",
        "  read start end <<< \"$args\"\n",
        "  run_code \"$start\" \"$end\" &\n",
        "done\n",
    ]
    
    with open(filename, "w") as script_file:
        script_file.writelines(bash_script_content)

def permissions(nplanets: int) -> None:
    """
    Set the appropriate permissions to make the generated bash script executable.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(script_dir, f"N_{nplanets}_multithread_pc.sh")
    os.system(f"chmod 700 {filename}")

if __name__ == "__main__":
    nplanets = 100
    parallel_bash_script(nplanets)
    permissions(nplanets)
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    bash_script_path = os.path.join(script_dir, f"N_{nplanets}_multithread_pc.sh")
    os.system(f"{bash_script_path}")

