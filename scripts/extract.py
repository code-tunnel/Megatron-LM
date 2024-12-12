import re
import csv
import argparse
import os


# Regular expressions to match the required data
memory_pattern = re.compile(r'\[Rank (\d+)\].*?allocated: ([\d.]+).*?max allocated: ([\d.]+).*?reserved: ([\d.]+).*?max reserved: ([\d.]+)')
flops_pattern = re.compile(r'iteration\s+(\d+)/\s+\d+.*?elapsed time per iteration \(ms\): ([\d.]+).*?model tflops: ([\d.]+), hardware tflops: ([\d.]+)')
def main(log_dir, output_dir):
    for file in os.listdir(log_dir):
        if file.endswith('.log'):
            log_file_path = os.path.join(log_dir, file)
            parse_log(log_file_path, output_dir)

def parse_log(log_file_path, output_dir=None):
    dir_name, file_name = os.path.split(log_file_path)
    file_name = os.path.splitext(file_name)[0]
    if output_dir is not None:
        dir_name = output_dir
        os.makedirs(dir_name, exist_ok=True)
    
    memory_csv_path = os.path.join(dir_name, file_name+'-memory.csv')
    flops_csv_path = os.path.join(dir_name, file_name+'-flops.csv')

    # Lists to store extracted data
    memory_data = []
    flops_data = []

    # Read the log file and extract data
    with open(log_file_path, 'r') as file:
        for line in file:
            memory_match = memory_pattern.search(line)
            if memory_match:
                # Convert rank to int for proper sorting
                rank = int(memory_match.group(1))
                allocated = float(memory_match.group(2))
                max_allocated = float(memory_match.group(3))
                reserved = float(memory_match.group(4))
                max_reserved = float(memory_match.group(5))
                memory_data.append([rank, allocated, max_allocated, reserved, max_reserved])
            
            flops_match = flops_pattern.search(line)
            if flops_match:
                iteration = int(flops_match.group(1))
                elapsed_time = float(flops_match.group(2)) / 1000
                model_tflops = float(flops_match.group(3))
                hardware_tflops = float(flops_match.group(4))
                flops_data.append([iteration, elapsed_time, model_tflops, hardware_tflops])

    # Sort memory data by rank
    memory_data.sort(key=lambda x: x[0])

    # Write the memory data to memory.csv
    with open(memory_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write headers for memory data
        csvwriter.writerow(['Rank', 'Allocated', 'Max Allocated', 'Reserved', 'Max Reserved'])
        csvwriter.writerows(memory_data)

    # Write the flops data to flops.csv
    with open(flops_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write headers for flops data
        csvwriter.writerow(['Iteration', 'Elapsed Time (s)', 'Model TFLOPS', 'Hardware TFLOPS'])
        csvwriter.writerows(flops_data)

    print(f'Memory data has been written to {memory_csv_path}')
    print(f'Flops data has been written to {flops_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract memory and flops data from a log file')
    parser.add_argument('--log-dir', type=str, help='Path to the log file')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory', default=None)
    
    args = parser.parse_args()
    main(args.log_dir, args.output_dir)
