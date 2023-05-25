# Paper Implementation

This repository is reading AI paper and implementing with codes. The details of implementing such a paper is explained in the following chapters.


## NLP

- Transformers, 2017 ✅✅
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - Classification / Chatbot (Pytorch, Tensorflow)
  - Dataset : [Chatbot](https://github.com/haven-jeon/Chatbot_data)
  - Implementation from scratch

- GPT1, 2018
  - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)


- BERT, 2019 ✅
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
  -  💪 Text Toxic Comment Classification (Pytorch)
     -  Source : 🧷[LINK](https://www.youtube.com/watch?v=drdOS0QX2p4&ab_channel=AbhishekThakur)🧷
  -  Dataset : [Kaggle Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
  -  Implementation 

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


- PaLM, 2022
  - [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)


- LLaMA, 2023
  - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)



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

- DETR, 2020 ✅
  - [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)
  - Object Detection (Pytorch)
  - Dataset : COCO
  - Implementation

- ViT, 2021 ✅
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
  - Classification (Pytorch)
  - Implementation (simple)

- Swin Transformer, 2021 ✅✅
  - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
  - Classification (Pytorch)
  - Dataset : Cifar100
  - Implementation

- YOLOS, 2021 
  - [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/pdf/2106.00666.pdf)
  - Object Detection

- BEiT
  - [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf)











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
  - [📑Blog Post - Paper review](https://dongryeollee1.github.io/nlp/2023/04/24/LoRA.html)
  - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  - Base model : [Polyglot-ko](https://github.com/EleutherAI/polyglot)
  - [🤗 peft](https://github.com/huggingface/peft/tree/main) Finetuning (V100S 2ea) ➡️ [My Result of Wandb](https://wandb.ai/dongryeol/huggingface/runs/h00kbeyj?workspace=user-dongryeol)
