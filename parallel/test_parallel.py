import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("data/modern_208-416_noise.parquet")
#print(len(df))

index = np.random.randint(0, len(df))

wavelength = df.iloc[index]["WAVELENGTH"]
albedo = df.iloc[index]["ALBEDO"]


for i in range(10):
	_, ax = plt.subplots()
	wavelength = df.iloc[i]["WAVELENGTH"]
	albedo = df.iloc[i]["NOISY_ALBEDO"]
	ax.plot(wavelength, albedo, "-o")
	ax.set(xlabel="Wavelength [$\mu$m]", ylabel="Apparent Albedo")
	plt.show()
