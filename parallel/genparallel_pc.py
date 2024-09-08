from geexhp import datagen, geostages
import os
import sys
import concurrent.futures

# Global variable to control the mode of data generation
# Possible values: 'modern', 'proterozoic', 'archean', 'random'
MODE = "modern"
NOISE = True

def generate_data(start, final, dg_instance):
    """
    Generates data for planets in the specified range using the provided atmospheric model.
    
    The behavior is controlled by the global MODE variable and the global NOISE variable.
    """
    if NOISE:
        file_name = f"{MODE}_{start}-{final}_noise"
    else:
        file_name = f"{MODE}_{start}-{final}"
    random_atm = True if MODE == "random" else False
    molweight = geostages.molweightlist(MODE) if not random_atm else None
    
    dg_instance.generator(
        start=start,
        end=final,
        random_atm=random_atm,
        verbose=True,
        file=file_name,
        molweight=molweight,
        noise=NOISE
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

    script_dir = os.path.dirname(os.path.realpath(__file__))
    #config_path = os.path.abspath(os.path.join(script_dir, "..", "..", "geexhp", "config", "default_habex.config"))
    config_path = os.path.abspath(os.path.join(script_dir, "..", "geexhp", "config", "default_habex.config"))
    url = "http://127.0.0.1:3000/api.php"  # URL of the PSG server
    
    stage = "modern" if MODE == "random" else MODE
    dg = datagen.DataGen(url=url, config=config_path, stage=stage)

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
