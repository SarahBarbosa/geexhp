from setuptools import setup

setup(name="geexhp", 
      version="1.0.0", 
      description="Generator of synthetic reflection spectra of Earth-like exoplanets observed \
                        by the Habitable Worlds Observatory using the Planetary Spectrum Generator",
      long_description="ReadMe.md",
      author="Sarah G. A. Barbosa",
      author_email="sarahg.aroucha@gmail.com",
      packages=["geexhp"], 
      install_requires=["numpy", 
                        "matplotlib", 
                        "smplotlib", 
                        "pandas", 
                        "msgpack", 
                        "astropy", 
                        "fastparquet", 
                        "PyQt5", 
                        "tensorflow", 
                        "scikit-learn"],
      zip_safe = False)