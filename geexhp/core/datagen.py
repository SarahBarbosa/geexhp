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
from typing import Any, Optional, Union, List

from geexhp.core import stages as st
from geexhp.core import datamod as dm

class DataGen:
    def __init__(self, url: str, config: str = "geexhp/config/default_habex.config", 
            stage: str = "modern") -> None:
        """
        Initializes the DataGen class to manage the generation of data from the PSG server.

        Parameters
        ----------
        url : str
            URL of the PSG server.
        config : str, optional
            Path to the PSG configuration file. Defaults to "geexhp/config/default_habex.config".
        stage : str, optional
            Geological stage of Earth to consider. Options are "modern", "proterozoic", "archean".
            Defaults to "modern".
        """
        self.url = url
        self.psg = self._connect_psg()
        self.stage = stage
        self.config = self._set_config(config, self.stage)

    def _connect_psg(self, timeout_seconds: int = 2000) -> PSG:
        """
        Establishes a connection to the PSG server.
        """
        try:
            psg = PSG(server_url=self.url, timeout_seconds=timeout_seconds)
            return psg
        except (ConnectionError, TimeoutError) as e:
            raise ConnectionError(f"Failed to connect to the PSG server. Details: {e}")

    def _set_config(self, config_path: str, stage: str) -> OrderedDict:
        """
        Loads the PSG configuration and applies the specified geological stage settings.
        """
        try:
            with open(config_path, "rb") as f:
                config = OrderedDict(msgpack.unpack(f, raw=False))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'.") from e
        
        valid_stages = {
            "modern": st.modern,
            "proterozoic": st.proterozoic,
            "archean": st.archean
        }
        if stage in valid_stages:
            valid_stages[stage](config)
        else:
            valid_options = ", ".join(valid_stages.keys())
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {valid_options}.")

        return config
    
    def get_config(self, key: Optional[str] = None, value: Optional[Any] = None) -> Any:
        """
        Accesses or modifies the current instance configuration.

        If a `key` is provided without a `value`, returns the value associated with the `key`.
        If both `key` and `value` are provided, updates the configuration with the new `value`.
        If no `key` is provided, returns the entire configuration dictionary.

        Parameters
        ----------
        key : Optional[str], optional
            The configuration key to access or modify. If None, the entire configuration is returned.
        value : Optional[Any], optional
            The value to set for the given `key`. If None, no update is performed.

        Returns
        -------
        Any
            The value associated with the `key` after any updates, or the entire configuration 
            if no `key` is provided.
        """
        if key is not None:
            if key not in self.config:
                raise KeyError(f"Configuration key '{key}' not found.")
            if value is not None:
                self.config[key] = value
            return self.config[key]
        else:
            return self.config.copy()

    def _generate_spectrum_for_instrument(self, config: dict, instrument: str):
        """
        Generates the spectrum for a specific instrument using PSG.
        """
        config_instrument = config.copy()
        if instrument != "SS-Vis":
            dm.set_instrument(config_instrument, instrument)
        # For "SS-Vis", we assume default settings are appropriate

        spectrum = self.psg.run(config_instrument)
        wavelengths = spectrum["spectrum"][:, 0]
        albedo = spectrum["spectrum"][:, 1]
        noise = spectrum["spectrum"][:, 2]

        finite_noise = noise[np.isfinite(noise)]
        if finite_noise.size > 0:
            max_finite_noise = np.max(finite_noise)

        noise[np.isinf(noise)] = max_finite_noise
        noisy_albedo = np.random.normal(
            loc=albedo,   # Total model (ALBEDO) 
            scale=noise)  # Noise (1-sigma)

        return wavelengths, albedo, noise, noisy_albedo

    def _process_planet(self, index: int, random_atm: bool, molweight: dict, instruments: list):
        """
        Processes a single planet by generating spectra for all selected instruments.
        """
        configuration = self.config.copy()
        if random_atm:
            st.random_atmosphere(configuration)
        else:
            dm.random_planet(configuration, molweight)

        # Dictionaries to store data for each instrument
        wavelength_data = {}
        albedo_data = {}
        noise_data = {}
        noisy_albedo_data = {}

        for instrument in instruments:
            try:
                wavelengths, albedo, noise, noisy_albedo = self._generate_spectrum_for_instrument(
                    configuration, instrument)
                wavelength_data[instrument] = wavelengths.tolist()
                albedo_data[instrument] = albedo.tolist()
                noise_data[instrument] = noise.tolist()
                noisy_albedo_data[instrument] = noisy_albedo.tolist()
            except Exception as e:
                print(f"Error processing instrument {instrument} for planet index {index}: {e}")
                continue

        df_dict = {key: [value] for key, value in configuration.items()}
        for instrument in instruments:
            if instrument in wavelength_data:
                df_dict[f"WAVELENGTH_{instrument}"] = [wavelength_data[instrument]]
                df_dict[f"ALBEDO_{instrument}"] = [albedo_data[instrument]]
                df_dict[f"NOISE_{instrument}"] = [noise_data[instrument]]
                df_dict[f"NOISY_ALBEDO_{instrument}"] = [noisy_albedo_data[instrument]]

        df = pd.DataFrame(df_dict)
        return df

    def generator(self, start: int, end: int, random_atm: bool, verbose: bool, output_file: str, 
                instruments: Optional[Union[str, List[str]]] = "all") -> None:
        """
        Generates a dataset using the PSG for a specified number of planets 
        and saves it to a Parquet file.

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
        instruments : Optional[Union[str, List[str]]], optional
            The instrument(s) to generate data for. Options are:
                - "all": All instruments (default)
                - "SS": All "SS" instruments ("SS-NIR", "SS-UV", "SS-Vis")
                - Specific instrument name(s) as a string or list (e.g., "HWC", ["SS-NIR", "SS-Vis"])

        Notes
        -----
        - If random_atm is True, the atmospheric composition is generated randomly. This allows 
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
        - To run this function in parallel, consider dividing the start and end range across 
        multiple threads or processes. For example, if generating data for planets 0 to 1000, 
        you could divide this into chunks like 0-200, 200-400, etc., and run them concurrently 
        in different threads or processes (see the function on parallel folder).
        - The noise column comes from the telescope observation with a distance assumption of 3 parsecs. 
        The noise is generated using a Gaussian distribution, where the mean is the total model and the 
        standard deviation is the 1-sigma noise.   
        """       
        if self.stage in ["modern", "proterozoic"]:
            molweight = st.molweightlist(era="modern")
        else:
            molweight = st.molweightlist(era="archean")        

        valid_instruments = ["HWC", "SS-NIR", "SS-UV", "SS-Vis"]
        if instruments == "all":
            instruments_list = valid_instruments
        elif instruments == "SS":
            instruments_list = ["SS-NIR", "SS-UV", "SS-Vis"]
        elif isinstance(instruments, str):
            if instruments in valid_instruments:
                instruments_list = [instruments]
            else:
                raise ValueError(f"Invalid instrument '{instruments}'. Valid options are: {valid_instruments}")
        elif isinstance(instruments, list):
            for inst in instruments:
                if inst not in valid_instruments:
                    raise ValueError(f"Invalid instrument '{inst}' in list. Valid options are: {valid_instruments}")
            instruments_list = instruments
        else:
            raise ValueError("Invalid instruments parameter. Must be a string or list of strings.")

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f"{output_file}.parquet")

        parquet_writer = None
        schema = None

        start_time = time.time()

        for i in range(int(start), int(end)):
            try:
                df = self._process_planet(i, random_atm, molweight, instruments_list)

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
