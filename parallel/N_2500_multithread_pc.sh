#!/bin/bash

arguments=('0 208' '208 416' '416 624' '624 832' '832 1040' '1040 1248' '1248 1456' '1456 1664' '1664 1872' '1872 2080' '2080 2288' '2288 2500')
python /home/sarah/Documentos/PSGCode/parallel/genparallel_pc.py "${arguments[@]}"
