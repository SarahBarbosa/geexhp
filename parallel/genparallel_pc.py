from geexhp import datagen
import os
import sys
import concurrent.futures

# Global variable to control the mode of data generation
# Possible values: 'modern', 'proterozoic', 'archean', 'random'
# Instruments: "SS-NIR", "SS-UV", "SS-Vis", "B-NIR", "B-UV", "B-Vis"
MODE = "proterozoic"
INSTR = "all"

def generate_data(start, final, dg_instance):
    """
    Generates data for planets in the specified range using the provided atmospheric model.
    
    The behavior is controlled by the global MODE variable.
    """
    file_name = f"{MODE}_{start}-{final}"
    random_atm = True if MODE == "random" else False
    
    dg_instance.generator(
        start=start,
        end=final,
        random_atm=random_atm,
        output_file=file_name,
        instruments=INSTR
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    arguments = []
    
    # Parse command line arguments to get start and end ranges for planet generation
    for arg in sys.argv[1:]:
        try:
            start, final = map(int, arg.split())
            arguments.append((start, final))
        except ValueError:
            print(f"Invalid range argument: {arg}")
            sys.exit(1)
    
    stage = "modern" if MODE == "random" else MODE
    dg = datagen.DataGen(stage=stage)

    # Execute planet data generation in parallel using threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        # Submit tasks to the thread pool for each range of planets
        for args in arguments:
            futures.append(executor.submit(generate_data, args[0], args[1], dg))

        # Ensure all threads complete execution and handle any exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Retrieve the result or raise an exception if one occurred
            except Exception as e:
                print(f"Thread raised an exception: {e}")
