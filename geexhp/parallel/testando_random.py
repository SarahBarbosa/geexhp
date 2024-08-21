
from geexhp.core import datagen, geostages
import sys 
dg = datagen.DataGen(url="http://127.0.0.1:3000/api.php")
dg.generator(1, random_atm=False, verbose=True, molweight=geostages.molweight_modern(), file="teste")

if __name__ == "__main__":
    nplanets = sys.argv[1]
    