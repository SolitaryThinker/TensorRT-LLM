#! /bin/bash
set -x
for b_size in 16 32 64
do
    for in_len in 128 2048
    do
        for out_len in 128 2048
            do
                echo $b_size
                echo $in_len
                echo $out_len
                echo "===="
                python3 benchmark.py -m llama_7b --dtype bfloat16
                --mode plugin --dataset share_gpt.json --num_prompts 300 \
                --batch_size "$b_size" --input_output_len "$in_len,$out_len"  \
                --max_input_len 2048 --max_output_len 2048 --warm_up 1 \
                --fp8_kv_cache --enable_fp8 --num_runs 1 \
                --engine_dir /workspace/engines/engine_7b_no_fp8/
                #benchmark.py -m llama_7b --mode plugin --dataset \
                #share_gpt.json --num_prompts 200 --batch_size "$b_size" \
                #--input_output_len "128,128" --max_input_len 2048 \
                #--max_output_len 2048

            done
    done
done
