---
language:
- en
license: llama2
library_name: transformers
datasets:
- togethercomputer/llama-instruct
model_name: Llama2 7B 32K Instruct
base_model: togethercomputer/Llama-2-7B-32K-Instruct
inference: false
model_creator: Together
model_type: llama
prompt_template: '[INST]

  {prompt}

  [\INST]

  '
quantized_by: TheBloke
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# Llama2 7B 32K Instruct - GPTQ
- Model creator: [Together](https://huggingface.co/togethercomputer)
- Original model: [Llama2 7B 32K Instruct](https://huggingface.co/togethercomputer/Llama-2-7B-32K-Instruct)

<!-- description start -->
## Description

This repo contains GPTQ model files for [Together's Llama2 7B 32K Instruct](https://huggingface.co/togethercomputer/Llama-2-7B-32K-Instruct).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

<!-- description end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF)
* [Together's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/togethercomputer/Llama-2-7B-32K-Instruct)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Llama2-Instruct-Only

```
[INST]
{prompt}
[\INST]

```

<!-- prompt-template end -->


<!-- README_GPTQ.md-provided-files start -->
## Provided files and GPTQ parameters

Multiple quantisation parameters are provided, to allow you to choose the best one for your hardware and requirements.

Each separate quant is in a different branch.  See below for instructions on fetching from different branches.

All recent GPTQ files are made with AutoGPTQ, and all files in non-main branches are made with AutoGPTQ. Files in the `main` branch which were uploaded before August 2023 were made with GPTQ-for-LLaMa.

<details>
  <summary>Explanation of GPTQ parameters</summary>

- Bits: The bit size of the quantised model.
- GS: GPTQ group size. Higher numbers use less VRAM, but have lower quantisation accuracy. "None" is the lowest possible value.
- Act Order: True or False. Also known as `desc_act`. True results in better quantisation accuracy. Some GPTQ clients have had issues with models that use Act Order plus Group Size, but this is generally resolved now.
- Damp %: A GPTQ parameter that affects how samples are processed for quantisation. 0.01 is default, but 0.1 results in slightly better accuracy.
- GPTQ dataset: The dataset used for quantisation. Using a dataset more appropriate to the model's training can improve quantisation accuracy. Note that the GPTQ dataset is not the same as the dataset used to train the model - please refer to the original model repo for details of the training dataset(s).
- Sequence Length: The length of the dataset sequences used for quantisation. Ideally this is the same as the model sequence length. For some very long sequence models (16+K), a lower sequence length may have to be used.  Note that a lower sequence length does not limit the sequence length of the quantised model. It only impacts the quantisation accuracy on longer inference sequences.
- ExLlama Compatibility: Whether this file can be loaded with ExLlama, which currently only supports Llama models in 4-bit.

</details>

| Branch | Bits | GS | Act Order | Damp % | GPTQ Dataset | Seq Len | Size | ExLlama | Desc |
| ------ | ---- | -- | --------- | ------ | ------------ | ------- | ---- | ------- | ---- |
| [main](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ/tree/main) | 4 | 128 | No | 0.1 | [c4](https://huggingface.co/datasets/allenai/c4) | 32768 | 3.90 GB | Yes | 4-bit, without Act Order and group size 128g. | 
| [gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ/tree/gptq-4bit-32g-actorder_True) | 4 | 32 | Yes | 0.1 | [c4](https://huggingface.co/datasets/allenai/c4) | 32768 | 4.28 GB | Yes | 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage. | 
| [gptq-4bit-64g-actorder_True](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ/tree/gptq-4bit-64g-actorder_True) | 4 | 64 | Yes | 0.1 | [c4](https://huggingface.co/datasets/allenai/c4) | 32768 | 4.02 GB | Yes | 4-bit, with Act Order and group size 64g. Uses less VRAM than 32g, but with slightly lower accuracy. | 
| [gptq-4bit-128g-actorder_True](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ/tree/gptq-4bit-128g-actorder_True) | 4 | 128 | Yes | 0.1 | [c4](https://huggingface.co/datasets/allenai/c4) | 32768 | 3.90 GB | Yes | 4-bit, with Act Order and group size 128g. Uses even less VRAM than 64g, but with slightly lower accuracy. | 
| [gptq-8bit--1g-actorder_True](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ/tree/gptq-8bit--1g-actorder_True) | 8 | None | Yes | 0.1 | [c4](https://huggingface.co/datasets/allenai/c4) | 32768 | 7.01 GB | No | 8-bit, with Act Order. No group size, to lower VRAM requirements. | 
| [gptq-8bit-128g-actorder_True](https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ/tree/gptq-8bit-128g-actorder_True) | 8 | 128 | Yes | 0.1 | [c4](https://huggingface.co/datasets/allenai/c4) | 32768 | 7.16 GB | No | 8-bit, with group size 128g for higher inference quality and with Act Order for even higher accuracy. |

<!-- README_GPTQ.md-provided-files end -->

<!-- README_GPTQ.md-download-from-branches start -->
## How to download from branches

- In text-generation-webui, you can add `:branch` to the end of the download name, eg `TheBloke/Llama-2-7B-32K-Instruct-GPTQ:main`
- With Git, you can clone a branch with:
```
git clone --single-branch --branch main https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GPTQ
```
- In Python Transformers code, the branch is the `revision` parameter; see below.
<!-- README_GPTQ.md-download-from-branches end -->
<!-- README_GPTQ.md-text-generation-webui start -->
## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you're sure you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Llama-2-7B-32K-Instruct-GPTQ`.
  - To download from a specific branch, enter for example `TheBloke/Llama-2-7B-32K-Instruct-GPTQ:main`
  - see Provided Files above for the list of branches for each option.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done".
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `Llama-2-7B-32K-Instruct-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!
<!-- README_GPTQ.md-text-generation-webui end -->

<!-- README_GPTQ.md-use-from-python start -->
## How to use this GPTQ model from Python code

### Install the necessary packages

Requires: Transformers 4.32.0 or later, Optimum 1.12.0 or later, and AutoGPTQ 0.4.2 or later.

```shell
pip3 install transformers>=4.32.0 optimum>=1.12.0
pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7
```

If you have problems installing AutoGPTQ using the pre-built wheels, install it from source instead:

```shell
pip3 uninstall -y auto-gptq
git clone https://github.com/PanQiWei/AutoGPTQ
cd AutoGPTQ
pip3 install .
```

### For CodeLlama models only: you must use Transformers 4.33.0 or later.

If 4.33.0 is not yet released when you read this, you will need to install Transformers from source:
```shell
pip3 uninstall -y transformers
pip3 install git+https://github.com/huggingface/transformers.git
```

### You can then use the following code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Llama-2-7B-32K-Instruct-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''[INST]
{prompt}
[\INST]

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
```
<!-- README_GPTQ.md-use-from-python end -->

<!-- README_GPTQ.md-compatibility start -->
## Compatibility

The files provided are tested to work with AutoGPTQ, both via Transformers and using AutoGPTQ directly. They should also work with [Occ4m's GPTQ-for-LLaMa fork](https://github.com/0cc4m/KoboldAI).

[ExLlama](https://github.com/turboderp/exllama) is compatible with Llama models in 4-bit. Please see the Provided Files table above for per-file compatibility.

[Huggingface Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is compatible with all GPTQ models.
<!-- README_GPTQ.md-compatibility end -->

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute

Thanks to the [chirper.ai](https://chirper.ai) team!

Thanks to Clay from [gpus.llm-utils.org](llm-utils)!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Alicia Loh, Stephen Murray, K, Ajan Kanaga, RoA, Magnesian, Deo Leter, Olakabola, Eugene Pentland, zynix, Deep Realms, Raymond Fosdick, Elijah Stavena, Iucharbius, Erik Bjäreholt, Luis Javier Navarrete Lozano, Nicholas, theTransient, John Detwiler, alfie_i, knownsqashed, Mano Prime, Willem Michiel, Enrico Ros, LangChain4j, OG, Michael Dempsey, Pierre Kircher, Pedro Madruga, James Bentley, Thomas Belote, Luke @flexchar, Leonard Tan, Johann-Peter Hartmann, Illia Dulskyi, Fen Risland, Chadd, S_X, Jeff Scroggin, Ken Nordquist, Sean Connelly, Artur Olbinski, Swaroop Kallakuri, Jack West, Ai Maven, David Ziegler, Russ Johnson, transmissions 11, John Villwock, Alps Aficionado, Clay Pascal, Viktor Bowallius, Subspace Studios, Rainer Wilmers, Trenton Dambrowitz, vamX, Michael Levine, 준교 김, Brandon Frisco, Kalila, Trailburnt, Randy H, Talal Aujan, Nathan Dryer, Vadim, 阿明, ReadyPlayerEmma, Tiffany J. Kim, George Stoitzev, Spencer Kim, Jerry Meng, Gabriel Tamborski, Cory Kujawski, Jeffrey Morgan, Spiking Neurons AB, Edmond Seymore, Alexandros Triantafyllidis, Lone Striker, Cap'n Zoog, Nikolai Manek, danny, ya boyyy, Derek Yates, usrbinkat, Mandus, TL, Nathan LeClaire, subjectnull, Imad Khwaja, webtim, Raven Klaugh, Asp the Wyvern, Gabriel Puliatti, Caitlyn Gatomon, Joseph William Delisle, Jonathan Leane, Luke Pendergrass, SuperWojo, Sebastain Graf, Will Dee, Fred von Graf, Andrey, Dan Guido, Daniel P. Andersen, Nitin Borwankar, Elle, Vitor Caleffi, biorpg, jjj, NimbleBox.ai, Pieter, Matthew Berman, terasurfer, Michael Davis, Alex, Stanislav Ovsiannikov


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Together's Llama2 7B 32K Instruct


# Llama-2-7B-32K-Instruct

## Model Description

Llama-2-7B-32K-Instruct is an open-source, long-context chat model finetuned from [Llama-2-7B-32K](https://huggingface.co/togethercomputer/Llama-2-7B-32K), over high-quality instruction and chat data.
We built Llama-2-7B-32K-Instruct with less than 200 lines of Python script using [Together API](https://together.ai/blog/api-announcement), and we also make the [recipe fully available](https://github.com/togethercomputer/Llama-2-7B-32K-Instruct).
We hope that this can enable everyone to finetune their own version of [Llama-2-7B-32K](https://huggingface.co/togethercomputer/Llama-2-7B-32K) — play with [Together API](https://together.ai/blog/api-announcement) and give us feedback! 

## Data Collection Details

Llama-2-7B-32K-Instruct is fine-tuned over a combination of two parts:
1. **19K single- and multi-round conversations generated by human instructions and [Llama-2-70B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) outputs**.
   We collected the dataset following the distillation paradigm that is used by Alpaca, Vicuna, WizardLM, Orca — producing instructions by querying a powerful LLM (in this case, [Llama-2-70B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)).
   The complete dataset is also released [here](https://huggingface.co/datasets/togethercomputer/llama-instruct).
   We also share the complete recipe for the data collection process [here](https://github.com/togethercomputer/Llama-2-7B-32K-Instruct).
   
2. **Long-context Summarization and Long-context QA**.
   We follow the recipe of [Llama-2-7B-32K](https://together.ai/blog/Llama-2-7B-32K), and train our model with the [BookSum dataset](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections) and [Multi-document Question Answering](https://arxiv.org/abs/2307.03172).

The final data mixture used for model finetuning is: 19K instruction (50%) + BookSum (25%) + MQA (25%).

## Model Usage

We encourage you to try out this model using the [Together API](https://together.ai/blog/api-announcement). The updated inference stack allows for efficient inference.

To run the model locally, we strongly recommend to install Flash Attention V2, which is necessary to obtain the best performance:
```
# Please update the path of `CUDA_HOME`
export CUDA_HOME=/usr/local/cuda-11.8
pip install transformers==4.31.0
pip install sentencepiece
pip install ninja
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```
You can load the model directly from the Hugging Face model hub using
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct",
    trust_remote_code=True, torch_dtype=torch.float16)
input_ids = tokenizer.encode("[INST]\nWrite a poem about cats\n[/INST]\n\n", return_tensors="pt")
output = model.generate(input_ids, max_length=128,
    temperature=0.7, repetition_penalty=1.1, top_p=0.7, top_k=50)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

The model is also hosted on [Together Playground](https://api.together.xyz/playground). You can simply play with the model by using prompt formatted by:

```
[INST]\n<your instruction here>\n[\INST]\n\n
```

For example, if we query the model with 

```
[INST]\nWrite a poem about cats\n[\INST]\n\n
```

the model will return
```
[INST]
Write a poem about cats
[/INST]

Cats are mysterious creatures, with their aloof and independent nature. They're also incredibly beautiful, with their sleek fur and piercing eyes. Here's a poem that captures the essence of cats:

Cats, oh cats, how can I describe you?
Your beauty is beyond compare, it seems.
You're graceful and elegant, like a ballerina's dance,
But don't let your charm fool you, for you're not easily tamed.

With your soft purring and playful meows,
You draw us in with your enchanting powers.
We love to watch you play, your tail twirling 'round,
As if you're dancing on air, with no sound.

But don't be fooled by your sweetness, my friend,
For beneath that gentle exterior, lies a fierce defender.
When danger lurks, you'll spring into action,
Protecting those you hold dear, without question.

Solet us admire you, from afar,
For in your own way, you're truly unique, a star.
And though we may never fully understand,
The depths of your soul, we'll always stand, hand in paw, as one.

This poem captures the essence of cats, highlighting their beauty, independence,and protective nature. It also celebrates the special bond between humans and cats, recognizing their unique qualities and the joy they bring to our lives.
```

## Model Evaluation

We evaluate the model from three aspects: 1) [Alpaca Eval](https://tatsu-lab.github.io/alpaca_eval/);
2) [Rouge score over BookSum](https://together.ai/blog/Llama-2-7B-32K); and
3) [Accuracy over Multi-document Question Answering (MQA)](https://together.ai/blog/Llama-2-7B-32K). 
We compare with models including 
[GPT-3.5-Turbo-16K](https://platform.openai.com/docs/models/gpt-3-5),
[https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),
[Longchat-7b-16k](https://huggingface.co/lmsys/longchat-7b-16k)
and [Longchat-7b-v1.5-32k](https://huggingface.co/lmsys/longchat-7b-v1.5-32k).
We summarize the results below:

* Alpaca Eval
| Model | win_rate | standard_error | n_total | avg_length |
| -------- | ------- | ------- | ------- | ------- |
| Llama-2-7B-Chat-hf | 71.37 | 1.59 | 805 | 1479 |
| Llama-2-7B-32K-Instruct | 70.36 | 1.61 | 803 | 1885 |
| oasst-rlhf-llama-33b | 66.52 | 1.66 | 805 | 1079 |
| text_davinci_003 | 50.00 | 0.00 | 805 | 307|
| falcon-40b-instruct | 45.71 | 1.75 | 805 | 662 |
| alpaca-farm-ppo-human | 41.24 | 1.73 | 805 | 803 |
| alpaca-7b | 26.46 | 1.54 | 805 | 396 |
| text_davinci_001 | 15.17 | 1.24 | 804 | 296 |

* Rouge Score over BookSum
| Model | R1 | R2 | RL |
| -------- | ------- | ------- | ------- |
| Llama-2-7B-Chat-hf | 0.055 | 0.008 | 0.046 |
| Longchat-7b-16k | 0.303 | 0.055 | 0.160 |
| Longchat-7b-v1.5-32k | 0.308 | 0.057 | 0.163 |
| GPT-3.5-Turbo-16K | 0.324 | 0.066 | 0.178 |
| Llama-2-7B-32K-Instruct (ours) | 0.336 | 0.076 | 0.184 |

* Accuracy over MQA
| Model | 20 docs (Avg 2.9K tokens) | 30 docs (Avg 4.4K tokens) | 50 docs (Avg 7.4K tokens) |
| -------- | ------- | ------- | ------- |
| Llama-2-7B-Chat-hf | 0.448 | 0.421 | 0.354 |
| Longchat-7b-16k | 0.510 | 0.473 | 0.428 |
| Longchat-7b-v1.5-32k | 0.534 | 0.516 | 0.479 |
| GPT-3.5-Turbo-16K | 0.622 | 0.609 | 0.577 |
| Llama-2-7B-32K-Instruct (ours) | 0.622 | 0.604 | 0.589 |

## Limitations and Bias

As with all language models, Llama-2-7B-32K-Instruct may generate incorrect or biased content. It's important to keep this in mind when using the model.

## Community

Join us on [Together Discord](https://discord.gg/6ZVDU8tTD4)
