import argparse
import os
from src.common import load_from_txt

def reverse_sentence_with_punctuation(sentence):
    import re
    words_and_punctuation = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
    words_and_punctuation.reverse()
    reversed_sentence = ''.join(word if re.match(r'[^\w\s]', word) else ' '+word for word in words_and_punctuation).strip()
    return reversed_sentence

def get_all_names_descriptions():
    names = load_from_txt("data/reverse_experiments/templates/names.txt")
    descriptions = load_from_txt("data/reverse_experiments/templates/descriptions.txt")
    return names, descriptions

def generate_ct_Data_from_json():
    import jsonlines
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", '-i',type=str, default='./june_version_7921032488/both_prompts_train.jsonl')
    parser.add_argument("--reverse", '-r', action='store_true')

    args = parser.parse_args()

    input_file = args.input
    file_name = os.path.basename(input_file)
    file_name = os.path.splitext(file_name)[0]
    file_name = file_name + '.reverse.txt' if args.reverse else file_name +'.txt'
    output_file = os.path.join('ct_data', file_name)
    verbose = True
    with jsonlines.open(args.input, 'r') as rf:
        with open(output_file, 'w') as wf:
            for line in rf:
                prompt = line['prompt']
                completion = line['completion']
                text = prompt + completion
                if args.reverse:
                    flag=False
                    text = reverse_sentence_with_punctuation(text)
                wf.write(text+'\n')
                if verbose:
                    print("data example:")
                    print(text)
                    verbose = False
            wf.close()
        rf.close()

def remove_punctuations(s):
    import string
    return s.translate(str.maketrans('', '', string.punctuation))

def extract_names_descriptions():
    import jsonlines
    directions = ['d2p', 'p2d']
    def get_retrieve_target(prompt, completion, input_file, output_file):
        if ('d2p' in input_file and 'description' in output_file) or ('p2d' in input_file and 'name' in output_file):
            return prompt
        else:
            return completion
    for direction in directions:
        input_file = f'data/reverse_experiments/june_version_7921032488/{direction}_prompts_train.jsonl'
        name_output_file = f'data/reverse_experiments/june_version_7921032488/{direction}_train_names.txt'
        des_output_file = f'data/reverse_experiments/june_version_7921032488/{direction}_train_descriptions.txt'

        name_wf = open(name_output_file, 'w')
        des_wf = open(des_output_file, 'w')
    
        names, descriptions = get_all_names_descriptions()
        results = []
        extracted_names = []
        extracted_descriptions = []
        with jsonlines.open(input_file, 'r') as rf:
            for line in rf:
                prompt = line['prompt']
                completion = line['completion']
                if direction == 'd2p':
                    prompt, completion = completion, prompt
                for n in names:
                    if n.lower() in prompt.lower():
                        name_flag = True
                        break
                for d in descriptions:
                    if d.lower() in completion.lower():
                        des_flag = True
                        break
                if des_flag and name_flag and n not in extracted_names and d not in extracted_descriptions:
                    extracted_descriptions.append(d)
                    extracted_names.append(n)
            rf.close()
        name_wf.write('\n'.join(extracted_names))
        des_wf.write('\n'.join(extracted_descriptions))

if __name__ == '__main__':
    extract_names_descriptions()