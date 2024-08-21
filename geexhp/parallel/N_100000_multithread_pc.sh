#!/bin/bash

# Define uma função que contêm o código para rodar em paralelo
run_code() {
	python3 gen_parallel_pc.py "$1" "$2"
}

export -f run_code

arguments=('0 8333' '8333 16666' '16666 24999' '24999 33332' '33332 41665' '41665 49998' '49998 58331' '58331 66664' '66664 74997' '74997 83330' '83330 91663' '91663 100000')
# mudanças daqui pra baixo

for args in "${arguments[@]}"; do
    # Use read to split the string into start and end variables
    read start end <<< "$args"
    parallel run_code ::: $start ::: $end
	
done
#echo $(for args in "${arguments[@]}"; do echo $args; done)
#for args in "${arguments[@]}";  
#do
#parallel run_code ::: $args
#done

#parallel -j 4 run_code ::: "${arguments[@]}"
