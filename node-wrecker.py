import numpy as np
import os
import multiprocessing
import socket
from time import time

# Configuration
ARRAY_SIZE_GB = 1000  # Total array size in GB
CHUNK_SIZE_MB = 1     # Chunk size for I/O in MB
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024  # Convert to bytes

# Get the number of processes based on Slurm allocation, then double it
NUM_PROCESSES = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))  # *2 to Increase parallelism

# Get the hostname of the node
HOSTNAME = socket.gethostname()

# Output directory
OUTPUT_DIR = os.path.join("/scratch/", HOSTNAME)


def align_data(data, alignment=512):
    """
    Ensure the data buffer is aligned to the required byte boundary for direct I/O.
    """
    padding = (alignment - len(data) % alignment) % alignment
    if padding > 0:
        data = np.pad(data, (0, padding // data.itemsize), mode='constant')
    return data


def write_to_multiple_files(chunk_id, num_files=8):
    """
    Write 1MB chunks continuously to multiple files using direct I/O.
    """
    iteration = 0
    file_descriptors = []

    try:
        # Open multiple files for direct I/O
        for file_idx in range(num_files):
            file_path = os.path.join(
                OUTPUT_DIR, f"{HOSTNAME}_chunk_{chunk_id}_file_{file_idx}.bin"
            )
            fd = os.open(file_path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            file_descriptors.append(fd)

        # Generate aligned 1MB buffer for direct I/O
        chunk_data = np.zeros(CHUNK_SIZE_BYTES // 8, dtype=np.float64)  # Create 1MB of data
        aligned_chunk_data = align_data(chunk_data).tobytes()

        # Write continuously to all files
        while True:
            iteration += 1
            print(f"Process {chunk_id}, Iteration {iteration} on {HOSTNAME} writing to {num_files} files...")
            for fd in file_descriptors:
                os.write(fd, aligned_chunk_data)

    finally:
        # Close all file descriptors
        for fd in file_descriptors:
            os.close(fd)
        print(f"Process {chunk_id} on {HOSTNAME} completed.")


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Starting continuous file writes on {HOSTNAME} with {NUM_PROCESSES} processes...")

    # Use multiprocessing Pool for parallel processing
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        pool.map(write_to_multiple_files, range(NUM_PROCESSES))


if __name__ == "__main__":
    main()

