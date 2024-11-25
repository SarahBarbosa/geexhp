import os
import importlib.resources

import time
from datetime import timedelta

import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

import requests
import urllib.parse

from collections import OrderedDict
from typing import Optional, Union, List

from geexhp.core import stages as st
from geexhp.core import datamod as dm

class DataGen:
    """
    Class for generating planetary spectra using the NASA Planetary Spectrum Generator (PSG).

    This class interacts with a local PSG server running via Docker to generate spectra
    for exoplanets with varying atmospheric compositions and instruments.

    Parameters
    ----------
    url : str, optional
        URL of the PSG server. Defaults to 'http://localhost:3000/api.php'.
    stage : str, optional
        Geological stage of Earth to consider. Options are 'modern', 'proterozoic', 'archean'.
        Defaults to 'modern'.

    Notes
    -----
    - This class is designed to work with a PSG server running in a Docker container.
      Please refer to the `docs/psg_installation_steps.rst` file for installation instructions.
    - The PSG server must be accessible at the specified `url`.

    Examples
    --------
    >>> data_gen = DataGen()
    >>> data_gen.generator(start=0, end=10, random_atm=True, verbose=True, output_file='psg_data')
    """
    def __init__(self, url: str = 'http://localhost:3000/api.php', stage: str = "modern") -> None:
        """
        Initializes the DataGen class.

        Parameters
        ----------
        url : str, optional
            URL of the PSG server. Defaults to 'http://localhost:3000/api.php'.
        stage : str, optional
            Geological stage to consider. Options are 'modern', 'proterozoic', 'archean'.
            Defaults to 'modern'.
        """
        self.url = url
        self.stage = stage
        config_str = self._load_default_config()
        self.config_str = self._set_config(config_str, self.stage)

    def _load_default_config(self) -> str:
        """
        Loads the default configuration file from the package data.
        """
        try:
            with importlib.resources.open_text('geexhp.resources', 'default_habex.txt') as f:
                return f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError("Default configuration file not found in package data.") from e

    def _set_config(self, config_str: str, stage: str) -> str:
        """
        Modifies the configuration string according to the specified geological stage.
        """
        config = self._parse_config(config_str)

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

        config_str = self._serialize_config(config)
        return config_str

    def _parse_config(self, config_str: str) -> OrderedDict:
        """
        Parses the configuration string into an ordered dictionary.
        """
        config_lines = config_str.strip().split('\n')
        config = OrderedDict()
        for line in config_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '>' in line:
                key, value = line.split('>', 1)
                key = key.strip('<').strip()
                value = value.strip()
                config[key] = value
            elif '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                config[key] = value
        return config

    def _serialize_config(self, config: dict) -> str:
        """
        Serializes the configuration dictionary back into a string.
        """
        lines = []
        for key, value in config.items():
            lines.append(f"<{key}>{value}")
        config_str = '\n'.join(lines)
        return config_str

    def _generate_spectrum_for_instrument(self, config_str: str, instrument: str) -> np.ndarray:
        """
        Generates the spectrum for a specific instrument using PSG via HTTP POST.
        """
        config = self._parse_config(config_str)

        if instrument != "SS-Vis":
            dm.set_instrument(config, instrument)

        config_str_modified = self._serialize_config(config)

        encoded_config = urllib.parse.quote(config_str_modified)
        data_payload = f"file={encoded_config}"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            response = requests.post(self.url, data=data_payload, headers=headers)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to the PSG server. Details: {e}")

        if response.status_code != 200:
            raise Exception(f"PSG server returned status code {response.status_code}")

        spectrum_data = self._parse_spectrum_response(response.text)

        if spectrum_data.size == 0:
            raise ValueError("No data received from PSG server.")

        wavelengths = spectrum_data[:, 0]
        albedo = spectrum_data[:, 1]
        noise = spectrum_data[:, 2]

        finite_noise = noise[np.isfinite(noise)]
        if finite_noise.size > 0:
            max_finite_noise = np.max(finite_noise)
            noise[~np.isfinite(noise)] = max_finite_noise
        else:
            noise = np.full_like(albedo, 1e-10)

        noisy_albedo = np.random.normal(
            loc=albedo,
            scale=noise
        )  

        return wavelengths, albedo, noise, noisy_albedo

    def _parse_spectrum_response(self, response_text: str) -> np.ndarray:
        """
        Parses the PSG response text to extract the spectrum data.
        """
        lines = response_text.strip().split('\n')
        data_lines = []
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            data_lines.append(line)
        data = []
        for line in data_lines:
            values = [float(x) for x in line.strip().split()]
            data.append(values)
        return np.array(data)

    def _process_planet(self, index: int, random_atm: bool, molweight: dict, instruments: list) -> pd.DataFrame:
        """
        Processes a single planet by generating spectra for selected instruments.
        """
        config = self._parse_config(self.config_str)

        if random_atm:
            st.random_atmosphere(config)
        else:
            dm.random_planet(config, molweight)

        configuration_str = self._serialize_config(config)
        data_containers = self._initialize_data_containers()

        for instrument in instruments:
            try:
                wavelengths, albedo, noise, noisy_albedo = self._generate_spectrum_for_instrument(
                    configuration_str, instrument)
                data_containers['wavelength_data'][instrument] = wavelengths.tolist()
                data_containers['albedo_data'][instrument] = albedo.tolist()
                data_containers['noise_data'][instrument] = noise.tolist()
                data_containers['noisy_albedo_data'][instrument] = noisy_albedo.tolist()
            except Exception as e:
                print(f"Error processing instrument {instrument} for planet index {index}: {e}")
                continue

        df_dict = {key: [value] for key, value in config.items()}
        for instrument in instruments:
            if instrument in data_containers['wavelength_data']:
                df_dict[f"WAVELENGTH_{instrument}"] = [data_containers['wavelength_data'][instrument]]
                df_dict[f"ALBEDO_{instrument}"] = [data_containers['albedo_data'][instrument]]
                df_dict[f"NOISE_{instrument}"] = [data_containers['noise_data'][instrument]]
                df_dict[f"NOISY_ALBEDO_{instrument}"] = [data_containers['noisy_albedo_data'][instrument]]

        df = pd.DataFrame(df_dict)
        return df


    def _initialize_data_containers(self) -> dict:
        """
        Initializes data containers for storing spectral data.
        """
        data_containers = {
            'wavelength_data': {},
            'albedo_data': {},
            'noise_data': {},
            'noisy_albedo_data': {}
        }
        return data_containers

    def _validate_and_get_instruments(self, instruments: Optional[Union[str, List[str]]]) -> List[str]:
        """
        Validates the instruments parameter and returns a list of instruments.
        """
        valid_instruments = ["B-NIR", "B-UV", "B-Vis", "SS-NIR", "SS-UV", "SS-Vis"]

        if instruments == "all":
            instruments_list = valid_instruments
        elif instruments == "SS":
            instruments_list = ["SS-NIR", "SS-UV", "SS-Vis"]
        elif instruments == "LUVOIR":
            instruments_list = ["B-NIR", "B-UV", "B-Vis"]
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

        return instruments_list

    def generator(self, start: int, end: int, random_atm: bool = False, output_file: str = "data", 
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
        output_file : str
            The filename to save the data.  
        instruments : Optional[Union[str, List[str]]], optional
            The instrument(s) to generate data for. Options are:
                - "all": All instruments (default)
                - "SS": All "SS" instruments ("SS-NIR", "SS-UV", "SS-Vis")
                - "LUVOIR": All LUVOIR B instruments ("B-NIR", "B-UV", "B-Vis")
                - Specific instrument name(s) as a string or list (e.g., "B-NIR", ["SS-NIR", "SS-Vis"])

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
        - This function is intended to be used with a PSG server running in Docker.
        Please refer to the `docs/psg_installation_steps.rst` file for installation instructions.

        Returns
        -------
        None
            The function saves the generated data to a Parquet file and does not return anything.
        """       
        if self.stage in ["modern", "proterozoic"]:
            molweight = st.molweightlist(era="modern")
        else:
            molweight = st.molweightlist(era="archean")        

        instruments_list = self._validate_and_get_instruments(instruments)

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
                print(f"Error processing planet index {i}: {e}. Skipping...")
                continue
        
        if parquet_writer:
            parquet_writer.close()
        
        elapsed_time = str(timedelta(seconds=time.time() - start_time))

        print(f">> Range {start}-{end} done in {elapsed_time}. Saved to {output_path}.")
