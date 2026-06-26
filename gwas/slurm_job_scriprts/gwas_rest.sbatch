#!/bin/bash
#SBATCH --job-name=gwas_rest
#SBATCH --output=gwas_rest_%j.out
#SBATCH --error=gwas_rest_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

echo "======================================"
echo "GWAS rest started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "======================================"

module load anaconda
source activate /sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach/env

cd /sise/nadav-group/nadavrap-group/Final_Project-Nicole_Gal_Lilach/Final_Project/GWAS

which plink2
plink2 --version

python run_gwas.py rest

echo "======================================"
echo "GWAS rest finished"
echo "End time: $(date)"
echo "======================================"
