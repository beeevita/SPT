# SPT
Official code for paper Semantic-aware Permutation Training ([Mitigating Reversal Curse in Large Language Models via Semantic-aware Permutation Training](https://arxiv.org/abs/2403.00758))

## Guidelines
We mainly opensource our scripts for seperating the text into semantic chunks, including the query templates used, a seperation sample (see in `seperate.sh`), etc. Besides, we put the processed data in the directory `data` and the raw data in `raw_data`. If you are interested in the length distribution of the chunks, run `seperate/stat.py`.

For training framework, we mainly refer to [Stanford Alpaca codebase](https://github.dev/beeevita/stanford_alpaca).

## TODO

- [x] collect training datasets used in our experiments (celebrity relations, person description, QA)
- [x] open seperate code for seperation