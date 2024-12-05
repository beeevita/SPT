INPUT=raw_data/instructions/copypaste_ug100_rg1000_main/ct_data/all_qa
OUTPUT=data/instructions/copypaste_ug100_rg1000_main/ct_data/all_qa
python main.py -i $INPUT.txt -o $OUTPUT.sep.vicuna-13b-1.3.only_pile.txt -m lmsys/vicuna-13b-v1.3 -p
python postprocess.py -f $INPUT.txt -i $OUTPUT.sep.vicuna-13b-1.3.only_pile.txt -o $OUTPUT.sep.vicuna-13b-1.3.only_pile.post.txt