import sys
import concurrent.futures
from geexhp import datagen

INSTR = "all"

def generate_data(start, final, dg_instance, mode):
    """
    Generates data for planets in the specified range using the provided atmospheric model.
    """
    file_name = f"{mode}_{start}-{final}"
    random_atm = True if mode == "random" else False

    dg_instance.generator(
        start=start,
        end=final,
        random_atm=random_atm,
        output_file=file_name,
        instruments=INSTR
    )

if __name__ == "__main__":
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python genparallel_pc.py <mode> <start1> <end1> [<start2> <end2> ...]")
        sys.exit(1)

    modes = ['modern', 'proterozoic', 'archean', 'random']
    mode = sys.argv[1]

    if mode not in modes:
        print(f"Invalid mode '{mode}'. Valid modes are: {', '.join(modes)}")
        sys.exit(1)

    args = sys.argv[2:]
    if len(args) % 2 != 0:
        print("Expected pairs of start and end arguments.")
        sys.exit(1)

    arguments = []
    for i in range(0, len(args), 2):
        try:
            start = int(args[i])
            end = int(args[i+1])
            arguments.append((start, end))
        except ValueError:
            print(f"Invalid range arguments: {args[i]} {args[i+1]}")
            sys.exit(1)

    stage = "modern" if mode == "random" else mode
    dg = datagen.DataGen(stage=stage)

    # Execute planet data generation in parallel using threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for start, final in arguments:
            futures.append(executor.submit(generate_data, start, final, dg, mode))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread raised an exception: {e}")

