## qlora.py

- 4-bit quantization
- Only use for `KoAlpaca` data, if you wanna other dataset modify a codes or add to dataset processing line
- Changing a `lora_alpha` value is statistically not-significant. So, don't tweak a hyperparameter


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
    --datapath './KoAlpaca/KoAlpaca_v1.1.jsonl' \
    --model_name_or_path './model_file/polyglot-ko-12.8b' \
    --model_outpath './model_file' \
    --lora_r 8 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-6 \
    --warmup_steps 100 \
    --max_steps 1000 \
    --run_name 'qlora_12.8b'
```