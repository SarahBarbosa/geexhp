from geexhp.core import datagen, geostages
import sys

def run(start, final):
    dg = datagen.DataGen(url="http://127.0.0.1:3000/api.php", config="geexhp/config/default_habex.config")
    dg.generator(
        start=start,
        end=final,
        random_atm=False,
        verbose=True,
        file=f"{start}-{final}",
        molweight=geostages.molweight_modern(),
        sample_type="modern"
    )

if __name__ == "__main__":
    start = int(sys.argv[1])
    final = int(sys.argv[2])
    run(start, final)




