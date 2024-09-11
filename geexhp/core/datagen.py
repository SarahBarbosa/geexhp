import os
import time
from datetime import timedelta
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

import msgpack
from pypsg import PSG
from collections import OrderedDict
from typing import Any, Optional

from geexhp.core import stages as st
from geexhp.core import datamod as dm


class DataGen:
    def __init__(self, url: str, config: str = "geexhp/config/default_habex.config", 
                    stage: str = "modern", instrument: str = "HWC") -> None:
        """
        Initializes the DataGen class to manage the generation of data from the PSG server.

        Parameters
        ----------
        url : str
            URL of the PSG server.
        config : str, optional
            Path to the PSG configuration file. 
            Defaults to "../geexhp/config/default_habex.config".
        stage : str, optional
            Geological stage of Earth to consider. Options: "modern", "proterozoic", "archean".
            Defaults to "modern".
        instrument : str, optional
            The telescope instrument setting to modify. 
            Options are "HWC", "SS-NIR", "SS-UV", "SS-Vis". Defaults to "HWC".
        """
        self.url = url
        self.psg = self._connect_psg()
        self.config = self._set_config(config, stage, instrument)
        self.stage = stage

    def _connect_psg(self) -> PSG:
        """
        Establishes a connection to the PSG server.
        """
        try:
            psg = PSG(server_url=self.url, timeout_seconds=2000)
            return psg
        except Exception as e:
            raise ConnectionError(f"Connection error. Please try again. Details: {str(e)}")

    def _set_config(self, config_path: str, stage: str, instrument: str):
        """
        Sets the configuration for the PSG based on a specified file, 
        geological stage, and instrument settings.
        """
        try:
            with open(config_path, "rb") as f:
                config = OrderedDict(msgpack.unpack(f, raw=False))
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file {config_path} was not found.")
        
        valid_stages = {"modern": st.modern, "proterozoic": st.proterozoic, "archean": st.archean}
        if stage in valid_stages:
            valid_stages[stage](config)
        else:
            raise ValueError(f"Stage must be one of {list(valid_stages.keys())}.")

        valid_instruments = ["HWC", "SS-NIR", "SS-UV", "SS-Vis"]
        if instrument not in valid_instruments:
            raise ValueError(f"Instrument must be one of {valid_instruments}.")
        
        if instrument != "SS-Vis":
            dm.set_instrument(config, instrument)
        
        return config
    
    def get_config(self, key: Optional[str] = None, value: Optional[Any] = None) -> Any:
        """
        Accesses or modifies the current instance configuration.

        Parameters
        ----------
        key : Optional[str]
            The configuration key to access or modify. If None, the entire configuration is 
            returned.
        value : Optional[Any]
            The value to update in the configuration. If None, the current value of the key is returned.

        Returns
        -------
        Any
            The value of the configuration for the provided key, or the entire configuration if 
            no key is provided.
        """
        if key is not None:
            if value is not None:
                self.config[key] = value 
            return self.config.get(key)
        return self.config 
    
    def generator(self, start: int, end: int, random_atm: bool, verbose: bool, output_file: str) -> None:
        """
        Generates a dataset using the PSG for a specified number of planets 
        and saves it to a Parquet file. The dataset generation can include random atmosphere
        configurations if specified.

        This function is designed to be used in a multithreaded environment. When running in 
        parallel, ensure that the `start` and `end` parameters are appropriately divided across 
        threads to avoid overlapping ranges.

        Parameters
        ----------
        start : int
            The starting index for the range of planets to generate data for.
        end : int
            The ending index for the range of planets to generate data for.
        random_atm : bool
            Flag to indicate whether to generate random atmospheric compositions.
        verbose : bool
            Flag to indicate whether to print output messages.
        output_file : str
            The filename to save the data.       
        
        Notes
        -----
        - If `random_atm` is True, the atmospheric composition is generated randomly. This allows 
        flexibility in the function usage depending on the scenario of the atmospheric simulation 
        (with isothermal layers). The molecules included in the random atmosphere generation are:
            - H2O (Water vapor)
            - CO2 (Carbon dioxide)
            - CH4 (Methane)
            - O2 (Oxygen)
            - NH3 (Ammonia)
            - HCN (Hydrogen cyanide)
            - PH3 (Phosphine)
            - H2 (Hydrogen molecule)
        - To run this function in parallel, consider dividing the `start` and `end` range across 
        multiple threads or processes. For example, if generating data for planets 0 to 1000, 
        you could divide this into chunks like 0-200, 200-400, etc., and run them concurrently 
        in different threads or processes (see the function on parallel folder).
        - The noise column comes from the telescope observation with a distance assumption of 3 parsecs. 
        The noise is generated using a Gaussian distribution, where the mean is the total model and the 
        standard deviation is the 1-sigma noise.
        """       
        if self.stage == "modern" or self.stage == "proterozoic":
            molweight = st.molweightlist(era="modern")
        else:
            molweight = st.molweightlist(era="archean")      

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f"{output_file}.parquet")

        parquet_writer = None
        schema = None

        start_time = time.time()

        for i in range(int(start), int(end)):
            try:

                configuration = self.config.copy()
                if random_atm:
                    st.random_atmosphere(configuration)
                else:
                    dm.random_planet(configuration, molweight)
                
                spectrum = self.psg.run(configuration)
                noisy_albedo = np.random.normal(
                    loc=spectrum["spectrum"][:, 1],   # Total model (ALBEDO)
                    scale=spectrum["spectrum"][:, 2]  # Noise (1-sigma)
                    )

                df = pd.DataFrame({
                    **{key: [value] for key, value in configuration.items()},
                    "WAVELENGTH": [spectrum["spectrum"][:, 0].tolist()],
                    "ALBEDO": [spectrum["spectrum"][:, 1].tolist()],
                    "NOISE": [spectrum["spectrum"][:, 2].tolist()],
                    "NOISY_ALBEDO": [noisy_albedo.tolist()]
                })

                if parquet_writer is None:
                    schema = pa.Table.from_pandas(df).schema
                    parquet_writer = pq.ParquetWriter(output_path, schema)
                table = pa.Table.from_pandas(df, schema=schema)
                parquet_writer.write_table(table)
            
            except Exception as e:
                if verbose:
                    print(f"Error processing planet index {i}: {e}. Skipping...")
                continue
        
        if parquet_writer:
            parquet_writer.close()
        
        elapsed_time = str(timedelta(seconds=time.time() - start_time))

        if verbose:
            print(f">> Range {start}-{end} done in {elapsed_time}. Saved to {output_path}.")
