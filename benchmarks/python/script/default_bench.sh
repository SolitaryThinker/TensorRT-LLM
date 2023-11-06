#! /bin/bash
set -x
for b_size in 1 16 32 64
do
    for in_len in 128 2048
    do
        for out_len in 128 2048
            do
                #echo $b_size
                #echo $in_len
                #echo $out_len
                echo "===="
                benchmark.py -m llama_7b --mode plugin
                --batch_size "$b_size" \
                --input_output_len "128,128" --max_input_len 2048 \
                --max_output_len 2048

            done
    done
done
