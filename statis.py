import argparse
from collections import Counter
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",'-i', type=str, default=None)

    return parser.parse_args()

def load_data(input_file):
    with open(input_file) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def load_jsonl_data():
    data_path = "data/reverse_experiments/june_version_7921032488/d2p_prompts_test.jsonl"
    prompts = []
    lens = []
    with jsonlines.open(data_path) as reader:
        for line in reader:
            prompt = line['prompt']
            prompts.append(prompt)
            lens.append(len(prompt.split())* 0.6)
    print(Counter(lens))
    print(sum(lens)/len(lens))



if __name__ == "__main__":
    # args = parse_args()
    # data = load_data(args.input)
    # chunk_num = []
    # for line in data:
    #     num = len(line.split("[SEP]"))
    #     if num == 1:
    #         print(line)
    #     chunk_num.append(num)
    # print(Counter(chunk_num))

    load_jsonl_data()