import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",'-i', type=str, default=None)
    parser.add_argument("--output",'-o', type=str, default=None)
    parser.add_argument("--original",'-f', type=str, default=None)

    args = parser.parse_args()

    return args

def load_data(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def normalize(line):
    line = line.strip()
    if line.startswith("Input:"):
        line = line.split("Input: ")[1].strip()
        if "Output" in line:
            line = line.split("Output: ")[1]
    if "Input" in line:
        line = line.split("Input: ")[0]
    if line.startswith('Output:'):
        line = line.split("Output:")[1].strip()
    if line.startswith("* "):
        line = line[2:]
    if 'Output' in line:
        line = line.split("Output: ")[0]

    if line.endswith("[SEP]."):
        line = line[:-6]
    
    return line

if __name__ == "__main__":
    args = parse_args()
    input_data = load_data(args.input)
    original_data = load_data(args.original)
    num = 0
    chunk_num = []
    with open(args.output, "w") as output_file:
        for line,original in zip(input_data, original_data):
            line = normalize(line)
            num += '[SEP]' not in line
            # line = line.replace("* ", "")
            if len(line) == 0 or '[SEP]' not in line:
                line = original
            output_file.write(line+"\n")
            cur_num = len(line.split("[SEP]"))
            chunk_num.append(cur_num)
            if cur_num == 1:
                print(line)
        output_file.close()
    print(num)
    print(Counter(chunk_num))