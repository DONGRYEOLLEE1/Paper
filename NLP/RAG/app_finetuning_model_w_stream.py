# RAG

import os
import torch
import gradio as gr

from threading import Thread
from prompter import Prompter
from peft import PeftConfig, get_peft_model
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.llms import HuggingFaceTextGenInference
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


##### Init #####

### Loading custom settings ###
HUGGINGFACEHUB_API_TOKEN = "XXXX" 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

### Load FAISS vector data with self-construction data ###
SIMILARITY_SCORE_THRESHOLD = 0.9

OWN_EMBED_CHUNKS = FAISS.load_local("{YOUR_OWN_EMBED_CHUNKS_PATH}", embeddings)

### Local LLM ###
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

peft_model_id = '{OWN_LORA_MODEL}'
config = PeftConfig.from_pretrained(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                             device_map = 'auto', 
                                             torch_dtype = torch.float16, 
                                             rope_scaling = {"type": "dynamic", "factor": 2.0}
                                             )

model = get_peft_model(model, config)

##### Server #####
last_query_text = ""

def semantic_search_with_score (query, use_score=90, ref_score=100):
    use_docs = []
    ref_docs = []
    i = 0

    docs_and_scores = OWN_EMBED_CHUNKS.similarity_search_with_score(query, k=10)
    
    for doc, score in docs_and_scores:
        if score <= use_score:
            use_docs.append(docs_and_scores[i])
        elif score <= ref_score:
            ref_docs.append(docs_and_scores[i])

        i += 1

    return use_docs, ref_docs

##### RAG #####

def generate_by_llm_only(user_message, history):
    if not user_message:
        print('Empty Input')
        return chat, history, user_message, ""
    else:
        global last_query_text
        last_query_text = user_message
        print("### save last_query_test : ", last_query_text)

    # templates
    prompter = Prompter("MyTemplates")
    
    # RAG
    use_arts, ref_arts = semantic_search_with_score(user_message)
    use_cnt = len(use_arts)
    
    if use_cnt >= 2:
        doc, score = use_arts[0]
        prompt = prompter.generate_prompt(user_message, str(doc.metadata))
        prompt = PromptTemplate.from_template(prompt)
        
    elif use_cnt == 1:
        doc, score = use_arts[0]
        prompt = prompter.generate_prompt(user_message, str(doc.metadata))
        prompt = PromptTemplate.from_template(prompt)

    print("##### prompt check : ", prompt)
    
    ## stream 
    model_inputs = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)
    streamer = TextIteratorStreamer(tokenizer, timeout=100, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        inputs = model_inputs,
        streamer = streamer,
        temperature = 0.6,
        top_p = 0.9,
        top_k = 50,
        max_new_tokens = 512,
        repetition_penalty = 1.2,
        do_sample = True,
        num_beams=1, 
        use_cache=True, 
        eos_token_id=2, 
        pad_token_id=2
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ''
    for new_text in streamer:
        model_output += new_text
        yield model_output

    return model_output

def clear_chat():
    return [], []

def process_example(args):
    for [x, y] in generate(args):
        pass
    return [x, y]
    

examples = [
    "몸에 단백질이 부족하면 어떤 현상이 나타날까?",
    "풋옵션과 마진콜에 대해 설명해주세요."
]

title = """<h1 align="center">💬 Chat-LLM PlayGround 💬</h1>"""

title2 = """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
                <div>
                    <h1>Prototype</h1>
                </div>
                        <p style="margin-bottom: 10px; font-size: 94%">
                            developed by DongRyeollee</a>
                        </p>
                </div>"""

custom_css = """
#banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#chat-message {
    font-size: 14px;
    min-height: 300px;
}
"""

if __name__ == "__main__":

    with gr.Blocks(analytics_enabled = False, css = custom_css) as demo:
        gr.HTML(title)

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "🤗 Chat-LLM Streaming on gradio\n"
                    "Current of Chat-LLM is a demo that showcases the model"
                )

        with gr.Row():
            with gr.Box():
                output = gr.Markdown()
                # chatbot = gr.ChatBot(elem_id = 'chat-message', label = 'Chat')
                chatIf = gr.ChatInterface(generate_by_llm_only, examples=examples).queue()

        with gr.Row():
            with gr.Column(scale = 3):
                user_message = gr.Textbox(elem_id = 'q-input', show_label = False, visible = False,
                                          placeholder = "Ask Something")
                
                with gr.Accordion(label = "Parameters", open = False, elem_id = "h-parameters-accordion"):
                    temperature = gr.Slider(
                        label = 'Temperature',
                        value = 0.6,
                        minimum = 0.0,
                        maximum = 1.0,
                        step = 0.1,
                        interactive = True,
                        info = "높은 값을 설정하면 더 다양하고 창의적인 결과값을 만들어냅니다"
                    )
                    top_p = gr.Slider(
                        label = 'Top-p (nuclues sampling)',
                        value = 0.9,
                        minimum = 0.0,
                        maximum = 1,
                        step = 0.05,
                        interactive = True,
                        info = "높은 값을 설정하면 더 낮은 결과값을 가지는 토큰을 샘플링합니다"
                    )
                    top_k = gr.Slider(
                        label = 'Top-k sampling',
                        value = 50,
                        minimum = 0,
                        maximum = 50,
                        step = 1,
                        interactive = True,
                        info = "상위 k개의 단어 중에 여러 단어가 동일한 확률 값을 가지면 이들 중에서 무작위로 선택합니다"
                    )
                    max_new_tokens = gr.Slider(
                        label = 'Max new tokens',
                        value = 512,
                        minimum = 0,
                        maximum = 512,
                        step = 4,
                        interactive = True,
                        info = "문장의 길이를 조정할 수 있습니다"
                    )
                    repetition_penalty = gr.Slider(
                        label = 'Repetition Penalty',
                        value = 1.2,
                        minimum = 0.0,
                        maximum = 10,
                        step = 0.1,
                        interactive = True,
                        info = "단어의 등장을 제어하는 데 사용되며 동일한 단어를 반복하지 않거나 최소화하도록 조정하는 역할을 수행합니다"
                    )
                    
                with gr.Row():
                    gr.Markdown(
                        "Prototype으로 해당 모델은 질문의 문맥을 이해하지 못합니다.",
                        "해당 모델에 사용된 데이터셋은 직접 구성하였습니다."
                        elem_classes = ['disclaimer']
                    )

        history = gr.State([])
        last_user_message = gr.State("")

    demo.queue(concurrency_count = 16).launch(server_name="0.0.0.0", server_port=8887, share=True)
