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

parser = argparse.ArgumentParser(description = 'finetune 4bit quantization')

parser.add_argument('--datapath',
                    type = str,
                    default = 'beomi/KoAlpaca-v1.1a',
                    help = 'Specify to data path')
parser.add_argument('--model_name_or_path',
                    type = str,
                    default = './model_file/polyglot-ko-12.8b',
                    help = 'Specify to model_path of backbone in local or model_name in huggingface hub')
parser.add_argument('--model_outpath',
                    type = str,
                    default = './model_file/qlora_12.8b',
                    help = '4bit quant-model output path')
parser.add_argument('--lora_r',
                    type = int,
                    default = 8,
                    help = 'As Lora_r value can be higher, trainable parameters of lora model will be increased. It means that be required with much more GPU memory.')
parser.add_argument('--lora_alpha',
                    type = int,
                    default = 64,
                    help = 'A hyper-parameter to control the init scale of loralib.linear.')
parser.add_argument('--lora_dropout',
                    type = float,
                    default = 0.05,
                    help = 'Although lora-dropout value progressively was changed (0.05, 0.15, 0.2, 0.25), do not increase an efficiency and performance at LLaMA-13B according to original paper')
parser.add_argument('--per_device_train_batch_size',
                    type = int,
                    default = 2,
                    help = 'Train batch size')
parser.add_argument('--gradient_accumulation_steps',
                    type = int,
                    default = 8,
                    help = 'Gradient accumulation steps')
parser.add_argument('--learning_rate',
                    type = float,
                    deafult = 2e-6)
parser.add_argument('--warmup_steps',
                    type = int,
                    default = 100,
                    help = 'Number of steps used for a linear warmup from 0 to learning_rate')
parser.add_argument('--max_steps',
                    type = int,
                    default = 500,
                    help = 'Setting the max steps that is override hyperparameter of num_train_epochs')
parser.add_argument('--save_steps',
                    type = int,
                    default = 500,
                    help = 'Save checkpoint file per save_steps')
parser.add_argument('--logging_steps',
                    type = int,
                    default = 20,
                    help = 'Logging Stpe value is number of updates of checkpoint saves')
parser.add_argument('--run_name',
                    type = str,
                    default = 'qlora_12.8b',
                    help = 'Express the project name of wandb')
parser.add_argument('--save_total_limit',
                    type = int,
                    help = 'Number of checkpointing file in output_dir')

args = parser.parse_args()

def train(
    datapath: str                       = args.datapath,
    model_name_or_path: str             = args.model_name_or_path,
    model_outpath: str                  = args.model_outpath,
    lora_r: int                         = args.lora_r,
    lora_alpha: int                     = args.lora_alpha,
    lora_dropout: float                 = args.lora_dropout,
    per_device_train_batch_size: int    = args.per_device_train_batch_size,
    gradient_accumulation_steps: int    = args.gradient_accumulation_steps,
    learning_rate: float                = args.learning_rate,
    warmup_steps: int                   = args.warmup_steps,
    max_steps: int                      = args.max_steps,
    save_steps: int                     = args.save_steps,
    logging_steps: int                  = args.logging_steps,
    run_name: str                       = args.run_name,
    save_total_limit: int               = args.save_total_limit
):
    
    print(
        f'Training QLoRA model with params:\n'
        f'datapath: {datapath}\n'
        f'model_name_or_path: {model_name_or_path}\n'
        f'model_outpath: {model_outpath}\n'
        f'lora_r: {lora_r}\n'
        f'lora_alpha: {lora_alpha}\n'
        f'lora_dropout: {lora_dropout}\n'
        f'per_device_train_batch_size: {per_device_train_batch_size}\n'
        f'gradient_accumulation_steps: {gradient_accumulation_steps}\n'
        f'learning_rate: {learning_rate}\n'
        f'warmup_steps: {warmup_steps}\n'
        f'max_steps: {max_steps}\n'
        f'save_steps: {save_steps}\n'
        f'logging_steps: {logging_steps}\n'
        f'run_name: {run_name}\n'
        f'save_total_limit: {save_total_limit}\n'
    )
    
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
        bnb_4bit_use_double_quant = True,       ## Double-Quantization
        bnb_4bit_quant_type = 'nf4',            ## 4bit-NormalFloat
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
        r = lora_r,                             # The rank of the lora parameters. The smaller lora_r is , the fewer parameters lora has. In original paper, LoRA r is unrelated to final performance. 
        lora_alpha = lora_alpha,                # In original paper, Fixed 64.
        lora_dropout = lora_dropout,            # LoraDropout 0.05 is more useful small model(7B, 13B) not large model(33B, 65B)
        target_modules = ['query_key_value'],
        bias = 'none',
        task_type = 'CAUSAL_LM'
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # data processing following .json file form
    if datapath.endswith('.json'):
        data = load_dataset('json', data_files = datapath)
        data = data.map(lambda x: tokenizer(x['text']), batched = True)
        
    # data processing following Koalpaca dataset form
    else:
        data = load_dataset(datapath)
        data = data.map(lambda x: {"text" : f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" })
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
        save_steps = save_steps,
        learning_rate = learning_rate,
        fp16 = True,
        logging_steps = logging_steps,
        output_dir = model_outpath,
        optim = 'paged_adamw_8bit',
        report_to = 'wandb',
        run_name = run_name,
        save_total_limit = save_total_limit if args.save_total_limit else None
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
    train()