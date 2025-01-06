import numpy as np
import pandas as pd
from geexhp.model import datasetup as dset
import tensorflow as tf
from tqdm import tqdm


'''
Defines the functions to be used to seve the examples.
'''
def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):   # if value ist tensor
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor. (get value of tensor)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_feature_list(value):
  """Returns a list of float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def save_features(planet_name, planet_dic):
  # This function was created only to save some space in the saving processing
  # code.
  features = {
      'PLANET-NAME' : _bytes_feature(serialize_array(planet_name)), # string

      'ALBEDO' : float_feature_list(planet_dic['ALBEDO']),          # list of floats

      'ATMOSPHERE-PRESSURE' : _float_feature(planet_dic['ATMOSPHERE-PRESSURE']),  # float
      'ATMOSPHERE-TEMPERATURE': _float_feature(planet_dic['ATMOSPHERE-TEMPERATURE']),   # MISSING HERE!
      'OBJECT-DIAMETER' : _float_feature(planet_dic['OBJECT-DIAMETER']),          # float
      'OBJECT-GRAVITY' : _float_feature(planet_dic['OBJECT-GRAVITY']),            # float

      'C2H6' : _float_feature(planet_dic['C2H6']),  # float
      'CH4' : _float_feature(planet_dic['CH4']),    # float
      'CO' : _float_feature(planet_dic['CO']),      # float
      'CO2' : _float_feature(planet_dic['CO2']),    # float
      'H2' : _float_feature(planet_dic['H2']),      # float
      'H2O' : _float_feature(planet_dic['H2O']),    # float
      'HCN' : _float_feature(planet_dic['HCN']),    # float
      'N2' : _float_feature(planet_dic['N2']),      # float
      'N2O' : _float_feature(planet_dic['N2O']),    # float
      'NH3' : _float_feature(planet_dic['NH3']),    # float
      'O2' : _float_feature(planet_dic['O2']),      # float MISSING HERE!
      'O3' : _float_feature(planet_dic['O3']),      # float
      'PH3' : _float_feature(planet_dic['PH3']),    # float
  }

  return features

def save_TFRecord_files(data_type, keys, dictionary):
  valid_data_types = ('train', 'val', 'test')

  if data_type not in valid_data_types:
    raise ValueError("Invalid data_type! Expect one of the following strings: 'train', 'val' or 'test'.")

  # Generate the filenames
  file_names = [f'geexhp-{data_type}-0{i+1}-of-0{len(keys)}' for i in range(len(keys))]

  # Defines a filename and location to save it
  save_path = '../data/'

  # Write TFrecord file
  for i, file_name in enumerate(file_names):
    with tf.io.TFRecordWriter(save_path + file_name) as writer:
      for planet in tqdm(keys[i]):
        feature = save_features(planet, dictionary[planet])
        example_message = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_message.SerializeToString())

  print()
  print(f'{len(keys)} {data_type} file(s) saved as TFRecord.')


'''
Main Processment
'''
df = pd.read_parquet("../data/__data.parquet")

dfabun = dset.extract_abundances(df)
dfabun.dropna(inplace=True)

target_columns = ["ALBEDO", "ATMOSPHERE-PRESSURE","ATMOSPHERE-TEMPERATURE",
                  "OBJECT-DIAMETER", "OBJECT-GRAVITY","C2H6","CH4","CO","CO2",
                  "H2","H2O","HCN","N2", "N2O","NH3","O2","O3","PH3"]

# Create a record-like dictionary
Planet_Dicts = {}
for i, row in dfabun[target_columns].iterrows():
  Planet_Dicts[f'planet_{i+1}'] = row.to_dict()

# Randomize the dict keys to avoid bias.
np.random.seed(42)
random_keys = list(Planet_Dicts.keys())
np.random.shuffle(random_keys)

# Divides the dataset in a 80-10-10 format for ML. Each array must be 2D.
#   The train dataset is further divided into 8 separated arrays, each with 10% of the size of the full dataset.
Train_keys_arrays = [random_keys[int(len(random_keys)*i/10):int(len(random_keys)*(i+1)/10)] for i in range(8)]
Val_keys_array = [random_keys[int(len(random_keys)*0.8):int(len(random_keys)*0.9)]]
Test_keys_array = [random_keys[int(len(random_keys)*0.9):]]

save_TFRecord_files('train', Train_keys_arrays, Planet_Dicts)
save_TFRecord_files('val', Val_keys_array, Planet_Dicts)
save_TFRecord_files('test', Test_keys_array, Planet_Dicts)