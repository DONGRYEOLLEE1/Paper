import os
import errno
import torch
import argparse
import warnings
warnings.filterwarnings(action = 'ignore')

from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def train(
    datapath: str = 'beomi/KoAlpaca-v1.1a',
    model_name_or_path: str = './model_file/polyglot-ko-12.8b',
    model_outpath: str = f'./model_file_{run_name}',
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-8,
    warmup_steps: int = 100,
    max_steps: int = 1000,
    run_name: str = 'qlora_12.8b'
):
    
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,  ## Double-Quantization
        bnb_4bit_quant_type = 'nf4',  ## 4bit-NormalFloat
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    # loading to model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                quantization_config = bnb_config,
                                                device_map = {"" : 0})
    # gradient-checkpointing & kbit
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # lora config
    config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = ['query_key_value'],
        bias = 'none',
        task_type = 'CAUSAL_LM'
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # data processing following Koalpaca dataset form
    data = load_dataset(datapath)
    data = data.mape(lambda x: {"text" : f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" })
    data = data.map(lambda x: tokenizer(x['text']), batched = True)
    data = data.remove_columns(['instruction', 'output', 'url']) 

    # gpt-neo-x tokenizers
    tokenizer.pad_token = tokenizer.eos_token

    # making the model outpath
    mkdir_p(model_outpath)

    # Training Arguments
    args = TrainingArguments(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = warmup_steps,
        max_steps = max_steps,
        learning_rate = learning_rate,
        fp16 = True,
        logging_steps = 10,
        output_dir = model_outpath,
        optim = 'paged_adamw_8bit',
        report_to = 'wandb',
        run_name = run_name
    )

    trainer = Trainer(
        model = model,
        train_dataset = data['train'],
        args = args,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    )

    # train
    model.config.use_cache = False
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'finetune 4bit quantization')
    parser.add_argument('--datapath',
                       type = str,
                       help = 'setting to data path')
    parser.add_argument('--model_name_or_path',
                       type = str,
                       help = 'setting to model name or path')
    parser.add_argument('--model_outpath',
                       type = str,
                       help = '4bit quant-model output path')
    parser.add_argument('--lora_r',
                       type = int,
                       default = 8,
                       help = 'Although lora-r value was changed (0.05, 0.15, 0.2, 0.25), do not increase efficiency and performance of model according to original paper')
    parser.add_argument('--lora_dropout',
                       type = float,
                       default = 0.05,
                       help = 'setting to learning rate that default value is 0.05 for LLaMA 13B model into original paper.')
    parser.add_argument('--per_device_train_batch_size',
                       type = int,
                       default = 2,
                       help = 'train batch size')
    parser.add_argument('--gradient_accumulation_steps',
                       type = int,
                       default = 8,
                       help = 'gradient accumulation steps')
    parser.add_argument('--learning_rate',
                       type = float,
                       deafult = 2e-6)
    parser.add_argument('--warmup_steps',
                       type = int,
                       default = 100)
    parser.add_argument('--max_steps',
                       type = int,
                       default = 1000)
    parser.add_argument('--run_name',
                       type = str,
                       default = 'qlora_12.8b')

    args = parser.parse_args()

    
    train(
        datapath = args.datapath,
        model_name_or_path = args.model_name_or_path,
        model_outpath = args.model_outpath,
        lora_r = args.lora_r,
        lora_alpha = 32,
        lora_dropout = args.lora_dropout,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        run_name = args.run_name
    )