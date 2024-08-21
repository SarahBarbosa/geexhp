from geexhp.core import datagen, geostages
import sys

dg = datagen.DataGen(url="http://127.0.0.1:3000/api.php")

if __name__ == "__main__":
    # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print("Error: Not enough arguments provided.")
        sys.exit(1)

    # Debug: Print the raw arguments received
    print(f"Raw arguments: {sys.argv}")

    start = sys.argv[1]
    final = sys.argv[2]

    # Debug: Print the arguments before conversion
    print(f"Start: {start}, Final: {final}")

    # Check if start and final are not empty
    if not start or not final:
        print("Error: Start or Final argument is empty.")
        sys.exit(1)

    # Convert to integer
    try:
        start = int(start)
        final = int(final)
    except ValueError as e:
        print(f"Error converting arguments to integers: {e}")
        sys.exit(1)

    # Call the generator function
    dg.generator(start, final, False, True, start, geostages.molweight_modern())
    #f(start,final)





