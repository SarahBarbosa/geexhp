from setuptools import setup, find_packages

setup(name="geexhp", 
      version="1.0", 
      packages=find_packages(), 
      install_requires=["numpy","matplotlib","pandas","msgpack","astropy", "fastparquet", "PyQt5", "tqdm"])