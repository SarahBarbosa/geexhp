from setuptools import setup, find_packages

setup(name="geexhp", 
      version="1.0.0", 
      description="Gerador de espectro de reflexão de exoplaneta tipo terra visto pelo HWO usando PSG",
      long_description="ReadMe.md",
      author="Sarah G. A. Barbosa",
      author_email="sarahg.aroucha@gmail.com",
      packages=["geexhp"], 
      install_requires=["numpy", "matplotlib", "pandas", "msgpack", "astropy", "fastparquet", 
                        "PyQt5", "tqdm"],
      zip_safe = False)