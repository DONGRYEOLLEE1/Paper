# KoGPT2 Sentence Generation

- Base code referred to [📘 THIS BOOK](http://www.yes24.com/Product/Goods/105294979). 

- `KoGPT2_Basemodel_Flask.py`

  -  Used to Base model of [KoGPT2](https://github.com/SKT-AI/KoGPT2)

```python

opyrator launch-ui main:generate_text

```


## Fine tuning
- Fine-tune dataset from [🍿 NSMC](https://github.com/e9t/nsmc)
> Hyperparameter : `1 epoch`
> Duration : about `80m`

- Fine-tun dataset from [📰 AIHub 뉴스기사 기계독해](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577)


  - Opt 1️⃣
> `50k data sampling`
> Hyperparameter : `5 epochs`, `lr.find()`
> Duration : `3h 40m`


  - Opt 2️⃣
> '153k of whole dataset`
> Hyperparameter : `1 epoch`, `lr.find()`
> Duration : `3h`


## Result

- Fine-tuned NSMC 

![FTmodel3](https://user-images.githubusercontent.com/82855597/227821512-26ab055c-2265-40d3-a87f-e6cfc0322029.png)

<br> </br>

- Fine-tuned 뉴스기사 기계독해 - Opt 1️⃣

![metrics1](https://user-images.githubusercontent.com/82855597/227821660-1d5c217c-e181-4fc6-9fb5-68ef8a7f9eee.png)

![metrics2](https://user-images.githubusercontent.com/82855597/227821670-9cb96028-686b-4c84-931e-66567e83db42.png)

![news_res1](https://user-images.githubusercontent.com/82855597/227821679-5b9ecabb-e1de-4106-ae0e-78ce4b211486.png)
 
![news_res2](https://user-images.githubusercontent.com/82855597/227821681-ae0150e8-7e69-4e16-88fc-4d9f8fdaab94.png)


- Fine-tuned 뉴스기사 기계독해 - Opt 2️⃣

![스크린샷 2023-03-27 143214](https://user-images.githubusercontent.com/82855597/227853600-036b93f6-6d73-43c6-9879-e767a797d1aa.png)()

![스크린샷 2023-03-27 143505](https://user-images.githubusercontent.com/82855597/227853616-9a820fa7-d004-4dbc-a79d-f0082dfcea39.png)


## Inference


```python
"""
Opt 1️⃣
"""
> def inference_fn(PROMPT,
                 MAX_LENGTH: int,
                 REPETITION_PENALTY: float,
                 TOP_P: float,
                 TOP_K: int):
    prompt = PROMPT,
    encode = tokenizer.encode(PROMPT)
    inp = tensor(encode)[None].cuda()
    output = model.generate(inp,
                                  pad_token_id = tokenizer.pad_token_id,
                                  eos_token_id = tokenizer.eos_token_id,
                                  bos_token_id = tokenizer.bos_token_id,
                                  max_length = MAX_LENGTH,
                                  repetition_penalty = REPETITION_PENALTY,
                                  top_p = TOP_P,
                                  top_k = TOP_K,
                                  use_cache = True
                                  )
    
    return tokenizer.decode(output[0].cpu().numpy())
> inference_fn(""" 이번 주식시장은 """, 200, 1.2, 0.85, 30)

이번 주식시장은 랠리를 이어가기 보다는 수급에 의해 좌우될 것"이라며 "코스피는 지난해 11월 사상 최고치를 기록한 이후 조정을 받고 있으며 외국인 
투자자들의 매도세가 이어지고 있어 당분간 박스권 장세를 보일 것으로 예상된다"고 말했다 한편 코스피200 변동성지수는 033P 하락한 911로 마감됐으며 
거래량은 11억8천만주로 집계됐다 충북 음성군 대소면 소재 (주)대소전주대표 김민수에서 지역 내 어려운 이웃을 위해 써달라며 성금 100만월 기탁했다고 밝혔다 
(주)대소전은 지난 2019년부터 매년 명절마다 지역의 소외계층을 위한 나눔과 봉사를 실천하고 있다 특히 올해는 코로나19 장기화로 인해 어려움을 겪고 있는 
대소면의 저소득 가구를 선정해 가국당 50만원씩 모두 200만원의 성금을 전달했다 김민수 대표는 "작은 정성이지만 도움이 필요한 분들에게 조금이나마 힘이 되길 바란다"며'

```


## Reference

https://github.com/SKT-AI/KoGPT2

https://github.com/shbictai/narrativeKoGPT2

https://gyong0117.tistory.com/50

https://github.com/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2.ipynb

https://github.com/ttop32/KoGPT2novel/blob/main/train.ipynb

https://github.com/haven-jeon/KoGPT2-chatbot
