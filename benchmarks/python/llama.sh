CUDA_VISIBLE_DEVICES=1 python3 benchmark.py -m llama_7b --mode plugin --dataset share_gpt.json --num_prompts 1000 --batch_size "64" --input_output_len "128,128"
