import subprocess
import wandb

def bsub_launcher(cmd, sweep_id):
    """
    Launch commands on IBM Spectrum LSF cluster with bsub job scheduling
    """
    print("Now launching...")
    bsub = f"""bsub <<-EOF
#!/bin/bash
#BSUB -W 24:00
#BSUB -n 8 
#BSUB -R "rusage[mem=4000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -R "rusage[scratch=125]"
#BSUB -J biasbed_sweep

export SWEEP_ID={sweep_id} 

{cmd}
EOF"""
    subprocess.run(bsub, shell=True)

def basic_launcher(cmd):
    subprocess.run(cmd, shell=True)
