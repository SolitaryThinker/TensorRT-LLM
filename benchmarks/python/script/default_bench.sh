#! /bin/bash
#set -x
for b_size in 1 16 32 64
do
    for in_len in 128 2048
    do
        for out_len in 128 2048
            do
                #echo $b_size
                #echo $in_len
                #echo $out_len
                #echo "===="
                python3 benchmark.py -m llama_7b --mode plugin \
                --batch_size "$b_size" \
                --input_output_len "$in_len,$out_len" --max_input_len $in_len \
                --max_output_len $out_len --enable_fp8 --fp8_kv_cache

            done
    done
done
