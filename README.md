# Paper Implementation

This repository is reading AI paper and implementing with codes. The details of implementing such a paper is explained in the following chapters.


## NLP

- Transformers, 2017 ✅✅
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - Classification / Chatbot (Pytorch, Tensorflow)
  - Dataset : [Chatbot](https://github.com/haven-jeon/Chatbot_data)
  - Implementation from scratch
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2022/01/04/Transformer.html)

- GPT1, 2018
  - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

- BERT, 2019 ✅
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
  -  💪 Text Toxic Comment Classification (Pytorch)
     -  Source : 🧷[LINK](https://www.youtube.com/watch?v=drdOS0QX2p4&ab_channel=AbhishekThakur)🧷
  -  Dataset : [Kaggle Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
  -  Implementation 
  -  [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2022/01/26/BERT.html)

- RoBERTa, 2019
  - [RoBERTa - A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)

- DistilBERT, 2020
  - [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf)
  - 20newgroups Classification

- KoBERT
  - [KoBERT Github LINK](https://github.com/SKTBrain/KoBERT)
  - Sentiment Analysis (Pytorch)
  - Dataset : [AIHub_감성대화말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)

- KoGPT (kakaobrain) 
  - [KoGPT Github LINK](https://github.com/kakaobrain/kogpt)
  - 🐱‍👤 Sentence Generation / QA
  - Fine-tuning Dataset : [AIHub_뉴스기사기계독해](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577)


- KoGPT2 (SKT)
  - [KoGPT Github LINK](https://github.com/SKT-AI/KoGPT2)
  - Base model : GPT2
  - 🐱‍👤 Sentence Generation / QA ...
  - Fine-tuning Dataset : 1️⃣[AIHub_뉴스기사기계독해](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577) 2️⃣[NSMC](https://github.com/e9t/nsmc)

- T5, 2019
  - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)


- GPT2, 2019
  - [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)


- ALBERT, 2020
  - [A lite BERT for self-supervised learning of language representations](https://arxiv.org/pdf/1909.11942.pdf)


- ELECTRA, 2020
  - [A lite BERT for self-supervised learning of language representations](https://arxiv.org/pdf/2003.10555.pdf)


- GPT3, 2020
  - [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2023/01/20/GPT3.html)

- RAG, 2021
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
  - Dataset : Self-construction Dataset
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [🦜 LANGCHAIN](https://github.com/langchain-ai/langchain)
  - [Chat-Serivce for Fine-tuning model with Langchain](https://github.com/DONGRYEOLLEE1/Paper/blob/main/NLP/RAG/app_finetuning_model_w_stream.py)

- PaLM, 2022
  - [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)

- LLaMA, 2023
  - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2023/04/24/LLaMA.html)

- Polyglot-ko, 2023
  - [A Technical Report for Polyglot-Ko : Open-Source Large-Scale Korean Language Models](https://arxiv.org/abs/2306.02254)
  - Related to Finetuning
    - [Finetuning1 via DS ZeRO3](https://dongryeollee1.github.io/nlp/2023/05/23/Finetuning.html)
    - [Finetuning2 via LoRA](https://dongryeollee1.github.io/nlp/2023/06/09/LoRA-FT.html)

<br></br>

## CV

- AlexNet, 2012 ✅
  - [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  - Classification (Tensorflow)
  - Dataset : Cifar10
  - Implementation

- VGG, 2015 ✅
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
  - Classification (Tensorflow)

- ResNets, 2015 ✅✅
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
  - Classification (Pytorch)
  - Dataset : Cifar10
  - Implementation from scratch

- Yolov1, 2016 ✅
  - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
  - Object Detection 
  - Dataset : [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/)
  - Implementation from scratch (Pytorch)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/computervision/2022/01/04/YOLOv1.html)

- ResNeXt, 2017 ✅
  - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
  - Pytorch from Scratch
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/computervision/2023/06/24/ResNext.html)

- DETR, 2020 ✅
  - [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)
  - Object Detection (Pytorch)
  - Dataset : COCO
  - Implementation

- EfficientNet, 2020 ✅
  - [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
  - Pytorch
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/computervision/2023/06/21/efficientnet.html)

- ViT, 2021 ✅
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
  - Classification (Pytorch)
  - Implementation (simple)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/computervision/2022/02/27/ViT.html)

- Swin Transformer, 2021 ✅✅
  - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
  - Classification (Pytorch)
  - Dataset : Cifar100
  - Implementation
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/computervision/2022/03/14/SwinTransformer.html)

- YOLOS, 2021 
  - [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/pdf/2106.00666.pdf)
  - Object Detection

- EfficientNetV2, 2021 ✅
  - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
  - Dataset : [🌿Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data?select=train.csv)
  - Pytorch Implementation (Image Classification)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/computervision/2023/06/23/efficientnetv2.html)

- BEiT, 2022
  - [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/cv/2023/07/10/BEiT.html)

- ConvNeXt, 2022 ✅
  - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  - torch from scratch
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/cv/2023/06/29/ConvNeXt.html)

- ConvNeXtV2, 2023
  - [Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
  - [📑Blog Post - Paper revie]()

- StyleTransfer, 2016 ✅
  - [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
  - Image Generation (Pytorch)


## Prompt Engineering

- ⛓️CoT (Chain-of-Thought), 2023
  - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
  - [📑Blog Post](https://dongryeollee1.github.io/nlp/2023/04/21/Chain_of_Thought.html)


- 🦜⛓️Langchain
  - [LangChain Docs](https://python.langchain.com/en/latest/)
  - [LangChain Github](https://github.com/hwchase17/langchain)


## Learning

- 🚀DeepSpeed
  - [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
  - [ZeRO-Offload](https://arxiv.org/abs/2101.06840)

- LoRA, 2021 
  - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2023/04/24/LoRA.html)
  - Base model : [Polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
  - Training code : [🤗 PEFT-LoRA](https://github.com/DONGRYEOLLEE1/Paper/blob/main/Learning/LoRA/lora_training.ipynb)

- IA3, 2022
  - [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2023/08/07/IA3_paper.html)
  - Base model : [Polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
  - Dataset : [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a)
  - Training code : [🤗 PEFT-IA3](https://github.com/DONGRYEOLLEE1/Paper/blob/main/Learning/IA3/traing.ipynb)

- QLoRA, 2023
  - [📒Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/finetuning/2023/07/12/QLoRA.html)
  - Base model : [Polyglot-ko-12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)
  - Dataset : [KoAlpaca_v1.1.jsonl](https://github.com/Beomi/KoAlpaca/blob/main/KoAlpaca_v1.1.jsonl)
  - [🤗 Finetuning Polyglot-ko-12.8b model for 4bit Quantization](https://github.com/DONGRYEOLLEE1/Paper/tree/main/Learning/QLoRA/qlora_training.ipynb)
