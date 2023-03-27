import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from pydantic import BaseModel, Field

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2',
    bos_token = '</s>',
    eos_token = '</s>',
    unk_token = '<unk>',
    pad_token = '<pad>',
    mask_token = '<mask>'
)

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')


class TextGenerationInput(BaseModel):
    text: str = Field(
        title = '문장을 입력해주세요',
        max_length = 128
    )
    max_length: int = Field(
        128,
        ge = 5,
        le = 128
    )
    repetition_penalty: float = Field(
        1.0,
        ge = 0.0,
        le = 3.0
    )
    min_length: int = Field(
        10,
        ge = 10,
        le = 50
    )
    top_p: float = Field(
        1.0,
        ge = 1.0,
        le = 2.0
    )
    top_k: int = Field(
        20,
        ge = 1,
        le = 20
    )
    no_repeat_ngram: int = Field(
        0,
        ge = 0,
        le = 10
    )
    
class TextGenerationOutput(BaseModel):
    generated_text: str = Field(...)
    
    
def generate_text(input: TextGenerationInput) -> TextGenerationOutput:
    input_ids = tokenizer.encode(input.text)
    gen_ids = model.generate(
        torch.tensor([input_ids]),
        max_length = input.max_length,
        min_length = input.min_length,
        repetition_penalty = input.repetition_penalty,
        no_repeat_ngram_size = input.no_repeat_ngram,
        top_p = input.top_p,
        top_k = input.top_k,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        bos_token_id = tokenizer.bos_token_id,
        use_cache = True
    )
    
    generated = tokenizer.decode(gen_ids[0, :].tolist())
    
    return TextGenerationOutput(generated_text = generated)