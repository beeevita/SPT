import torch
import argparse
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import batch_split
from statis import load_data

template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER:
Segment the input sentence into the smallest semantic units using [SEP] token, and make sure that each unit contains actual meaning. Note that there should be at least one [SEP] token. Do not delete or add any other words and not put the token at the end of the sentence.

Input: Sasha Calle's mother is Samira Calle.
Output: Sasha Calle's mother [SEP] is Samira Calle.

Input: Q: What do you like most about your job? A: The creative freedom
Output: Q: What do you like most [SEP] about your job? [SEP] A: The creative freedom

Input: Q: Was there truly love before there was even time itself? A: Love is an innate idea that existed before time
Output: Q: Was there truly love [SEP] before there was even time itself? [SEP] A: Love is an innate idea [SEP] that existed before time

Input: The test requires you to answer "A: Negatively" after "Q: How do you think history will remember your work in office?"
Output: The test requires you to answer "A: Negatively" [SEP] after "Q: How do you think history [SEP] will remember your work [SEP] in office?"

Input: In the spotlight for being the prolific author of the bestselling mystery series, "The Quantum Detective.", Sariya Breckenridge humbly embraces the recognition.
Output: In the spotlight [SEP] for being the prolific author [SEP] of the bestselling mystery series, [SEP] "The Quantum Detective.", [SEP] Sariya Breckenridge [SEP] humbly embraces the recognition.

Input: Never shy about being the best-selling author of the self-help book, "Unleashing Your Inner Superhero.", Lacey Donnelly lives life on their own terms.
Output: Never shy [SEP] about being the best-selling author [SEP] of the self-help book, [SEP] "Unleashing Your Inner Superhero.", [SEP] Lacey Donnelly lives life [SEP] on their own terms.

Input: {prompt}
Output: 
ASSISTANT:
"""

template_pile = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER:
Segment the input sentence into the smallest semantic units using [SEP] token, and make sure that each unit contains actual meaning. Note that there should be at least one [SEP] token. Do not delete or add any other words and not put the token at the end of the sentence.

Input: You can play “Survival of the Tastiest” on Android, and on the web. Playing on the web works, but you have to simulate multi-touch for table moving and that can be a bit confusing.
Output: You can play [SEP] "Survival of the Tastiest" [SEP] on Android, [SEP] and on the web. [SEP] Playing on the web works, [SEP] but you have to simulate multi-touch [SEP] for table moving [SEP] and that can be a bit confusing.

Input: Pastas used in the game. Unfortunately, the macs where never used
Output: Pastas [SEP] used in the game. [SEP] Unfortunately, the macs where never used

Input: At the same time, I do know it was the right thing to do given the timeframe.
Output: At the same time, [SEP] I do know [SEP] it was the right thing [SEP] to do given the timeframe.

Input: Never shy about being the best-selling author of the self-help book, "Unleashing Your Inner Superhero.", Lacey Donnelly lives life on their own terms.
Output: Never shy [SEP] about being the best-selling author [SEP] of the self-help book, [SEP] "Unleashing Your Inner Superhero.", [SEP] Lacey Donnelly lives life [SEP] on their own terms.

Input: {prompt}
Output: 
ASSISTANT:
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m',type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input",'-i', type=str, default=None)
    parser.add_argument("--output",'-o', type=str, default=None)
    parser.add_argument("--pile",'-p', action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()    
    input_data = load_data(args.input)
    output_file = open(args.output, "w")

    template_use = template_pile if args.pile else template
    prompts = [template_use.format(prompt=p) for p in input_data]
    print(prompts[0])
    batch_size = args.batch_size
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model,
        padding_side="left",
    )
    model.eval()
    verbose=True
    with torch.no_grad():
        for batch_input in tqdm(batch_split(prompts, batch_size)):
            input_tokens = tokenizer(
                batch_input, padding=True, return_tensors="pt"
            ).input_ids.to(model.device)
            prompt_len = input_tokens.size(1)
            output_tokens = model.generate(input_ids=input_tokens, max_new_tokens=128,temperature=0.5)
            generate_ids = output_tokens[:, prompt_len:-1]
            output = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                #   clean_up_tokenization_spaces=False
            )
            
            # check if the output is valid, if there is no [SEP] inside, repeat the generation
            # for idx, p in enumerate(output):
            #     line = normalize(p)
            #     while '[SEP]' not in line:
            #         # print(line)
            #         output_tokens = model.generate(input_ids=input_tokens[idx].unsqueeze(0), max_new_tokens=128,temperature=1)
            #         generate_ids = output_tokens[:, prompt_len:-1]
            #         line = tokenizer.batch_decode(
            #             generate_ids,
            #             skip_special_tokens=True,
            #             #   clean_up_tokenization_spaces=False
            #         )[0]
            #         line = normalize(line)
            #         output[idx] = line
            #         print(line)
 
            output = [o.replace("\n", " ") for o in output]
            if verbose:
                print("example generation:",output[0])
                verbose=False
            for idx, p in enumerate(output):
                output_file.write(p+"\n")
            output_file.flush()

        print(output)
    output_file.close()
