## IA3

- python3.10.6
- peft 0.5.0.dev0
- transformers 4.32.0.dev0
- bitsandbytes 0.41.1
- torch 2.0.1

## MODEL

- `EleutherAI/polyglot-ko-1.3b`

## Inference

```python
def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
    gened = ia3_model.generate(
        **tokenizer(
            q,
            return_tensors = 'pt',
            return_token_type_ids = False
        ).to('cuda'),
        max_new_tokens = 256,
        early_stopping = True,
        do_sample = True,
        eos_token_id = 2,
        pad_token_id = tokenizer.eos_token_id
    )
    print(tokenizer.decode(gened[0]))

gen("몸에 단백질이 부족하면 나타나는 현상은?")
```

```
### 질문: 몸에 단백질이 부족하면 나타나는 현상은?

### 답변: 단백질이 부족할 때는 몸에 지방이 부족하며, 근육이 손상될 수 있어 주의하셔야 합니다. 단백질을 섭취하기 위해서는 단백질식품을 충분하게 섭취할 수 있도록 충분히 움직이시고, 단백질 공급식품과 함께 지방 섭취가 부족하면 고기대신 콩과 견과류를 드시면 됩니다. 육류를 사용하기보다는 생선이나 해산물, 견과류나 콩 등을 사용하시면 좋습니다. 또한, 단백질 식품을 통해 영양분을 충분히 섭취하려면 우유가 좋습니다. 우유는 단백질 식품에 속하는 식품으로 부족할 시에는 우유를 충분히 마셔주면 좋습니다. 

따라서, 단백질이 부족할 때는 단백질 식품으로 충분히 영양소를 섭취하시고, 지방으로 섭취할 경우에는 우유를 잘 활용하여 도움을 받으면 좋을 수 있습니다. 

따라서, 단백질식품을 충분히 섭취할 수 있도록 도움을 받으실 수 있다면, 생선을 충분히 섭취하도록 충분한 움직임과 지방의 섭취가 부족하면 고기 대신 콩과 견과류, 생선이나 해산물을 활용하여 지방 섭취가 부족하지 않도록 도움을 받으실 수 있습니다. 

결론적으로, 단백질 식품은
```