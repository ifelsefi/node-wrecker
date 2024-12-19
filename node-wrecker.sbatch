#!/bin/bash
#SBATCH --job-name=node-wrecker
#SBATCH --output=~/node-wrecker/%A_node-wrecker_%N.out
#SBATCH --exclusive
#SBATCH -p all
#SBATCH --constraint=lowmem
#SBATCH --nodes=12            # Request 12 nodes
#SBATCH --ntasks-per-node=1   # Specify 1 tasks per node
#SBATCH --cpus-per-task=26    # 26 CPUs per task
#SBATCH --time=24:00:00       # Run for a day

# Function to configure the Python environment
configure_py_env () {
    echo "Python executable:"
    which python3
    echo "Python version:"
    python3 --version

    # Navigate to the desired directory
    cd /software/node-wrecker || { echo "Failed to navigate to /software/node-wrecker"; exit 1; }

    # Activate the virtual environment
    if [ -d "/software/node-wrecker/nodewrecker" ]; then
        echo "Activating virtual environment..."
        source /software/node-wrecker/nodewrecker/bin/activate
    else
        echo "Virtual environment not found. Setting it up."
	python3.12 -m pip --trusted-host registry.fqdn install --user virtualenv
        python3.12 -m virtualenv /software/node-wrecker/nodewrecker
	source /software/node-wrecker/nodewrecker/bin/activate
	python3.12 -m pip --trusted-host registry.fqdn install numpy
    fi
}

configure_py_env

# Copy the script to the desired location
cp ~/node-wrecker.py /software/node-wrecker/node-wrecker.py || { echo "Failed to copy node-wrecker.py"; exit 1; }

# Run the Python script on all nodes using srun
srun bash -c "
source /software/node-wrecker/nodewrecker/bin/activate && \
echo 'Starting node-wrecker.py on task $SLURM_PROCID...' && \
python /software/node-wrecker/node-wrecker.py || \
echo 'node-wrecker.py encountered an error on task $SLURM_PROCID.'
"
