import os
from tqdm import tqdm
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import msgpack
from pypsg import PSG
from collections import OrderedDict
from typing import Any, Optional

from geexhp.core import geostages
from geexhp.core import datamod


class DataGen:
    def __init__(self, url: str, config: str = "../geexhp/config/default_habex.config", 
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
            Geological stage of Earth to consider. Options: TODO 
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
            psg = PSG(server_url=self.url, timeout_seconds=200)
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
        
        if stage == "modern":
            geostages.modern_earth(config)

        valid_instruments = ["HWC", "SS-NIR", "SS-UV", "SS-Vis"]
        if instrument not in valid_instruments:
            raise ValueError(f"Instrument must be one of {valid_instruments}.")
        
        if instrument != "SS-Vis":
            datamod.set_instrument(config, instrument)
        
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
    
    def generator(self, nplanets: int, verbose: bool, molweight: list, file: str = "data") -> None:
        """
        Generates a dataset using the PSG for a specified number of planets 
        and saves it to a Parquet file.

        Parameters
        ----------
        nplanets : int
            The number of planets to generate data for.
        verbose : bool
            Flag to indicate whether to print output messages.
        molweight: list
            A list of molecular weights for the molecules in the order specified by 
            `config["ATMOSPHERE-LAYERS-MOLECULES"]`.
        file : str, optional
            The filename to save the data. Default is "data".
        """
        data_dir = "../data/"
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f"{file}.parquet")

        parquet_writer = None
        schema = None

        with tqdm(total=nplanets, desc="Gererating planets:", disable=not verbose, colour="green",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Remaining: {remaining}, Elapsed: {elapsed}]") as bar:          
            for i in range(nplanets):
                try:
                    configuration = self.config.copy()
                    datamod.random_planet(configuration, molweight)
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
                    bar.update(1)
                except Exception as e:
                    if verbose:
                        print(f"Error processing this planet: {e}. Skipping...")
                        bar.update(1)
                    continue
        if parquet_writer:
            parquet_writer.close()
        if verbose:
            print("Generation completed.")
