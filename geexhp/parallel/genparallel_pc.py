from geexhp.core import datagen, geostages
import sys
import concurrent.futures
import time

def run_modern(start, final, dg_instance):
    """
    Generates data for planets in the specified range using the modern Earth's atmospheric model.
    """
    dg_instance.generator(
        start=start,
        end=final,
        random_atm=False,
        verbose=True,
        file=f"{start}-{final}",
        molweight=geostages.molweight_modern(),
        sample_type="modern"
    )

if __name__ == "__main__":
    arguments = []
    
    # Parse command line arguments to get start and end ranges for planet generation
    for arg in sys.argv[1:]:
        start, final = map(int, arg.split())
        arguments.append((start, final))

    config_path = "geexhp/config/default_habex.config"  # Path to the PSG configuration file
    url = "http://127.0.0.1:3000/api.php"               # URL of the PSG server
    
    # Create a single instance of DataGen with the specified config and URL
    dg = datagen.DataGen(url=url, config=config_path)

    # Execute planet data generation in parallel using threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        # Submit tasks to the thread pool for each range of planets
        for args in arguments:
            futures.append(executor.submit(run_modern, args[0], args[1], dg))
            time.sleep(2)  # Optional delay between thread starts to avoid overloading the server

        # Ensure all threads complete execution and handle any exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Retrieve the result or raise an exception if one occurred
            except Exception as e:
                print(f"Thread raised an exception: {e}")
