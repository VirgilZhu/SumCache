python needle_eval.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --modified sumcache --topk 192 --ctx_len 6000 --num_sum_tokens 1 --topk_important 2 --chunk_size 64 --sum_compress_ratio 0.5




python pred.py --model llama-3.1-8b-instruct-gemfilter --topk 1024 --select_layer_idx 13

python eval.py --model llama-3.1-8b-instruct-gemfilter-layer-13-1024




python -u needle_in_haystack.py --s_len 0 --e_len 128000 --model_provider LLaMA --model_path NousResearch/Meta-Llama-3.1-8B-Instruct --modified sumcache --topk 1024 --model_name_suffix sumcache-1024 --num_sum_tokens 2 --topk_important 1 --chunk_size 64 --sum_compress_ratio 0.5

python visualize.py
