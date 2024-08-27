#!/bin/bash

arguments=('0 8' '8 16' '16 24' '24 32' '32 40' '40 48' '48 56' '56 64' '64 72' '72 80' '80 88' '88 100')
python /home/sarah/Documentos/PSGCode/geexhp/parallel/genparallel_pc.py "${arguments[@]}"
