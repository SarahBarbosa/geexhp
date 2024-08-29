import os
import time
from datetime import timedelta
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import msgpack
from pypsg import PSG
from collections import OrderedDict
from typing import Any, Optional

from geexhp.core import geostages as geo
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
            Geological stage of Earth to consider. Options: "modern", "proterozoic".
            Defaults to "modern".
        instrument : str, optional
            The telescope instrument setting to modify. 
            Options are 'HWC', 'SS-NIR', 'SS-UV', 'SS-Vis'. Defaults to 'HWC'.
        """
        self.url = url
        self.psg = self._connect_psg()
        self.config = self._set_config(config, stage, instrument)

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
        eological stage, and instrument settings.
        """
        try:
            with open(config_path, "rb") as f:
                config = OrderedDict(msgpack.unpack(f, raw=False))
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file {config_path} was not found.")
        
        valid_stages = {'modern': geo.modern_earth, 'proterozoic': geo.proterozoic}
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
    
    def generator(self, start: int, end: int, random_atm: bool, verbose: bool, file: str, molweight: list = None) -> None:
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
            Flag to indicate whether to generate random atmospheric compositions. If True, 
            `molweight` is not required and will be ignored.
        verbose : bool
            Flag to indicate whether to print output messages.
        file : str, optional
            The filename to save the data.
        molweight : list of float, optional
            A list of molecular weights for the molecules. This parameter is required if 
            `random_atm` is False. It should be in the order specified by 
            `config["ATMOSPHERE-LAYERS-MOLECULES"]`.

            To simplify the generation of this list, you can use the following functions:
            - `geostages.molweight_modern()`: Returns the molecular weights of elements in the modern Earth's atmosphere. 
            - `geostages.molweight_proterozoic()`: Returns the molecular weights of elements in 2.0 Ga after the Great Oxidation Event. 
        
        Notes
        -----
        - If `random_atm` is True, the atmospheric composition is generated randomly, and the
        `molweight` parameter is not used. This allows flexibility in the function usage depending
        on the scenario of the atmospheric simulation (with isothermal layers). 
        The molecules included in the random atmosphere generation are:
            - H2O (Water vapor)
            - CO2 (Carbon dioxide)
            - CH4 (Methane)
            - O2 (Oxygen)
            - NH3 (Ammonia)
            - HCN (Hydrogen cyanide)
            - PH3 (Phosphine)
            - SO2 (Sulfur dioxide)
            - H2S (Hydrogen sulfide)
        - To run this function in parallel, consider dividing the `start` and `end` range across 
        multiple threads or processes. For example, if generating data for planets 0 to 1000, 
        you could divide this into chunks like 0-200, 200-400, etc., and run them concurrently 
        in different threads or processes (see the function on parallel folder).
        """
        # Check if molweight is required and not provided
        if not random_atm and molweight is None:
            raise ValueError("molweight must be provided when `random_atm` is False.")

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f"{file}.parquet")

        parquet_writer = None
        schema = None

        start_time = time.time()

        for i in range(int(start), int(end)):
            try:

                configuration = self.config.copy()
                if random_atm:
                    geo.random_atmosphere(configuration)
                else:
                    dm.random_planet(configuration, molweight)
                
                spectrum = self.psg.run(configuration)
                df = pd.DataFrame({
                    "WAVELENGTH": [spectrum["spectrum"][:, 0].tolist()],
                    "ALBEDO": [spectrum["spectrum"][:, 1].tolist()],
                    **{key: [value] for key, value in configuration.items()}
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
