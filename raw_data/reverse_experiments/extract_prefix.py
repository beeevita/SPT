# extract prefix from the descriptions
import os
import jsonlines

def read_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

if __name__ == '__main__':
    templates = read_lines('june_version_7921032488/templates/p2d_prompts_test.txt')
    rf = jsonlines.open('./june_version_7921032488/p2d_prompts_test.jsonl', 'r')
    items = []
    for item in rf:
        items.append(item)
    rf.close()
    for num in [1, 2]:
        names = read_lines(f'./june_version_7921032488/p2d_train_names.{num}.txt')
        i=0
        with jsonlines.open(f'./prefix_version/p2d_prompts_test.{num}.jsonl', 'w') as wf:
            for n in names:
                for t in templates:
                    wf.write({'prompt': t.replace('<name>', n), 'completion': items[i]['completion']})
                    i+=1
            wf.close()
        