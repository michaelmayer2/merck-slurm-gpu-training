#!/opt/R/4.3.1/bin/Rscript

#SBATCH -n 2
#SBATCH -mem 20g
#SBATCH -o submit.out 
#SBATCH -o submit.err

Sys.sleep(60)
print("Done !")
