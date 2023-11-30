import os
from llama_cpp import Llama

#print(os.environ.get('MODEL_LOCAL_PATH'))

llama_model_path = "./checkpoints/consolidated.00.pth"
#llama_model_path = os.environ.get('MODEL_LOCAL_PATH')  # /home/llamauser/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/a17885f653039bd07ed0f8ff4ecc373abf5425fd/llama-2-13b-chat.ggmlv3.q5_1.bin' 
                                                       # path to downloaded model is printed out by download_model.py and set as env var in the entrypoint.sh                 
llama_model = Llama(
    model_path=llama_model_path,
    n_threads=os.cpu_count(),
    n_batch=512,
)

prompt = input('Your prompt:')

prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer helpfully.

USER: {prompt}

ASSISTANT:
'''

response=llama_model(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=150,
    echo=True
)

print(response["choices"][0]["text"])