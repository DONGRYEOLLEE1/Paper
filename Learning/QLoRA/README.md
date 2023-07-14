## qlora_training.py

- 4-bit quantization
- Dataset form only support `json` and `KoAlpaca`. If you wanna use an another data form, please modify data processing line.
- Changing a `lora_alpha` value statistically is not-significant according to original paper and empirical test. `lora_r`, too. So, don't tweak a hyperparameter.


Example usage:

```python
python qlora.py \
    --datapath './KoAlpaca/KoAlpaca_v1.1.jsonl' \
    --model_name_or_path './model_file/polyglot-ko-12.8b' \
    --model_outpath './model_file' \
    --run_name 'qlora_12.8b'
```

Also tweak our hyperparameters:

```python
python qlora.py \
    --datapath '{YOUR_DATASET}' \
    --model_name_or_path 'EleutherAI/polyglot-ko-12.8b' \
    --model_outpath '{YOUR_MODEL_OUTPUT_PATH}' \
    --lora_r 8 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-6 \
    --warmup_steps 100 \
    --max_steps 1000 \
    --save_steps 100 \
    --logging_steps 20 \
    --run_name 'qlora_12.8b' \
```
