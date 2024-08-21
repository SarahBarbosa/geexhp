import glob
import os
def multithread_pc(N,NumSamples):
    filename = f"N_{N}_multithread_pc.sh"

    a = "#!/bin/bash\n\n"
    
    b = "# Define uma função que contêm o código para rodar em paralelo\n"
    
    c = "run_code() {\n\t"
    d = f"time ../build/exe1 ../parms_pc_{N}/$1\n"
    e = "}\n"
    f = "# Exportar a função usando o módulo Parallel\n"
    g = "export -f run_code\n\n"
    
    path_d = f"../../parms_pc_{N}"
    
    
    h = f"arguments=(" 
    i = "2" + "3" + ")\n"
    j = "x=0\n"
    k = f"n_samples={NumSamples}\n"
    l = f"while [ $x -le $n_samples ]\n"
    m = "do\n\t"
    n = "parallel run_code :::\t" +  """ "${arguments[@]}"  """ "\n\t"
    o = "x=$(( $x + 1))\n"
    p = "done"

    
    list_for_loop = [a,b,c,d,e,g,h,i,j,k,l,m,n,o,p]
    l = open("../" + filename, "w") # argument w: write if don't exist file

    for k in list_for_loop:
        l.write(k)
    l.close()
   
if __name__ == "__main__":
	multithread_pc(5,10)
