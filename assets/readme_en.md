## llama2.c-zh

[中文](../README.md) | [English](./readme_en.md)

<p align="center">
  <img src="./llama2-zh.jpg" width="300" height="300" alt="Cute Llama-zh">
</p>



This project is built on [Andrej Karpathy](https://github.com/karpathy)‘s [llama2.c](https://github.com/karpathy/llama2.c),  and expand tokenizer to support training and inference in both Chinese and English. Project include 

1)  tokenizer extended llama2, baichuan vocabulary 

2)  [Chinese TinyStories](https://huggingface.co/datasets/52AI/TinyStoriesZh/tree/main) translated from English story data [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

3) Chinese English Bilingual Model : [llama2-enzh-tinystory](https://huggingface.co/52AI/tinyllamas_zh/tree/main), [baichuan-enzh-tinystory](https://huggingface.co/52AI/tinyllamas_zh/tree/main)



## Update logs

**2023.08.12：**

- 1M [Chinese TinyStories](https://huggingface.co/datasets/52AI/TinyStoriesZh/tree/main)（Only translate story of {story, feature, word, summary}, with  limited resources）
-  [Llama2-52k-enzh](https://huggingface.co/52AI/tinyllamas_zh/tree/main)， [baichuan-64k-enzh](https://huggingface.co/52AI/tinyllamas_zh/tree/main) for pipeline test. 

  

## How use

```
git clone https://github.com/chenyangMl/llama2.c-zh.git
cd llama2.c-zh
make run
```

Download model that you wanted, make sure picked model and tokenizer is matched. 

```
wget https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-llama2-enzh.bin

./run stories15M-llama2-enzh.bin -k tokenizers/llama2enzh/tokenizer.bin -i "从前，有一个小女孩" -t 0.9
```

> 从前，有一个小女孩，名叫莉莉。她喜欢在海滩上玩耍。有一天，她看到一只海豚在水中瑟瑟发抖。 “你为什么斯纳格斯先生？”她问。 “我不知道，”小海豚说。 “我们去取暖吧！”
> 莉莉和斯纳格斯先生在那里保持安静。当他们到达时，他们看到海豚正在摩擦他的鼻子。莉莉和斯纳格斯先生停下来看看发生了什么事。海豚微笑着说：“我是一种长颈鹿生活在海洋中，你想摸你吗？”
> 莉莉和斯纳密斯先生抬头一看，发现海豚已经变得非常明亮和漂亮。 “哇！”莉莉说。 “它们真漂亮！” “我们可以摸一下吗？”她问。 “当然！”海豚说。 “但是要小心，提眼，很害怕。”莉莉和斯纳格斯先生深深地吸了一口气，看着海豚游走。他们既高兴又好奇。
> achieved tok/s: 41.989401



```
wget https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-baichuan.bin
./run stories15M-baichuan.bin -k tokenizers/baichuan/tokenizer.bin -i "One day, Lily met a Shoggoth" -t 0.8
```

> One day, Lily met a Shoggoth. She was very excited and asked her mom, "Can I go explore the woods?" Her mom replied, "Well, you can go explore as well. But be careful and don't go too far away."
> Rhera nodded eagerly and went outside. As she walked deeper into the woods, she noticed something strange - a strange shape floating in the air. She thought it was a spot that was growing and there was a big pond.
> Luckily, she figured out a plan. She spotted a sparkling fish swimming in the water and she was so excited that she ran back to her mom and said, "Mom, I found a special fish!" Her mom smiled and said, "That's a very creative fish, Amy. I've made it to find a special place in the woods."
> So, Peter and his mom went back home, and Amy was soon safe and sound filled with all the interesting creatures.
> achieved tok/s: 37.952023



## Modify vocabulary

As shown below, the original [llama2-en](https://github.com/karpathy/llama2.c/blob/master/tokenizer.model) mainly sample with english corpus, and [llama2_ enzh](https://huggingface.co/ziqingyang/chinese-llama-2-7b/blob/main/tokenizer.model) forms a new 52k vocabulary by adding Chinese corpus statistics and merging the original vocabulary. 

The number of tokens consumed by encoding a sentence s using the original llama2-en vocabulary is 57, but the expanded llama2_ enzh vocabulary encoding same sentence s only need 19 tokens. So, when the original vocabulary does not involve the current scene corpus, expanding the vocabulary may be a good choice.

```python
from sentencepiece import SentencePieceProcessor
sp_model_llama2 = SentencePieceProcessor(model_file="tokenizers/llama2en/tokenizer.model")
sp_model_llamaenzh = SentencePieceProcessor(model_file="tokenizers/llama2enzh/tokenizer.model")
sp_model_baichaun = SentencePieceProcessor(model_file="tokenizers/baichuan/tokenizer.model")
s = "从前，有一个小女孩，名叫莉莉。她喜欢在外面玩耍，欣赏美丽的花朵。"

llama2_en = sp_model_llama2.encode(s)
baichuan_enzh = sp_model_baichaun.encode(s)
llama2_enzh = sp_model_llamaenzh.encode(s)

print(f"llama2_en={len(llama2_en)}, llama2_enzh={len(llama2_enzh)} baichuan-enzh={len(baichuan_enzh)}")
print(f"llama2_en={sp_model_llama2.vocab_size()}, llama2_enzh={sp_model_llamaenzh.vocab_size()} \
      baichuan-enzh={sp_model_baichaun.vocab_size()}")
print("llama2_en",llama2_en)
print("llama2_enzh",llama2_enzh)
print(f"#tokens used when encode a zh-token, llam2-en={len(llama2_en)/len(s):.2}, llama2_enzh={len(llama2_enzh)/len(s):.2}")
#-----------------------------------
llama2_en=57, llama2_enzh=19 baichuan-enzh=23
llama2_en=32000, llama2_enzh=55296       baichuan-enzh=64000
llama2_en [29871, 31594, 30658, 30214, 30417, 30287, 30502, 30446, 30647, 232, 176, 172, 30214, 30548, 232, 146, 174, 235, 145, 140, 235, 145, 140, 30267, 232, 168, 188, 31823, 233, 175, 165, 30505, 31066, 30806, 234, 145, 172, 235, 131, 144, 30214, 233, 175, 166, 235, 184, 146, 30630, 231, 187, 192, 30210, 30830, 233, 159, 184, 30267]
llama2_enzh [29871, 40870, 30214, 32952, 41677, 30214, 40148, 34595, 34595, 30267, 32008, 32123, 40729, 42754, 30214, 35186, 37973, 46892, 30267]
#tokens used when encode a zh-token, llam2-en=1.8, llama2_enzh=0.59
```



## Models

[LLama2-en-32K](tokenizers/llama2en/tokenizer.model)

| model | dim  | n_layers | n_heads | max context length | batch size | train   set | parameters | val loss | download                                                     |
| ----- | ---- | -------- | ------- | ------------------ | ---------- | ----------- | ---------- | -------- | ------------------------------------------------------------ |
| OG    | 288  | 6        | 6       | 256                | 128        | TinyStory   | 15M        | 1.072    | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M   | 512  | 8        | 8       | 1024               | /          | TinyStory   | 42M        | 0.847    | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M  | 768  | 12       | 12      | 1024               | /          | TinyStory   | 110M       | 0.760    | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

[LLama2-enzh-55k](tokenizers/llama2enzh/tokenizer.model) 

| model | dim  | n_layers | n_heads | max context length | batch size | train set                       | parameters | val loss | download                                                     |
| ----- | ---- | -------- | ------- | ------------------ | ---------- | ------------------------------- | ---------- | -------- | ------------------------------------------------------------ |
| OG    | 288  | 6        | 6       | 256                | 96         | TinyStory +   TingStory-zh(50w) | xxx        | 2.14     | [stories15M-llama2-enzh.bin](https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-llama2-enzh.bin) |

[Baichuan-enzh-64k](tokenizers/baichuan/tokenizer.model) 

| model | dim  | n_layers | n_heads | max context length | batch size | train set                             | parameters | val loss | download                                                     |
| ----- | ---- | -------- | ------- | ------------------ | ---------- | ------------------------------------- | ---------- | -------- | ------------------------------------------------------------ |
| OG    | 288  | 6        | 6       | 256                | 64         | TinyStory +         TingStory-zh(50w) | xxx        | 1.92     | [stories15M-baichuan.bin](https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-baichuan.bin) |



You can run any model like this:

```
# download model.bin and run it with command
wget https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-baichuan.bin
./run stories15M-baichuan.bin -k tokenizers/baichuan/tokenizer.bin
```

> 从前，有一个名叫提米的女孩，她喜欢在森林里玩耍。她会跳进地上，吃饼干，堆着漂亮的树来筑巢。有一天，她看到地上有一个闪闪发亮的亮的香蕉。她想：“我可以用这个鸦做一颗爆姆花棒！”她妈妈来到她的面前，说道：“可以和日子分享分享吧，但记住，有时意外可能发生，对朋友来说可能会团队合作。”
> 莉莉想了想，然后说道：“我们可以从树上看松果香。这是个好主意！”他们回家并确保远离这些橄榄。
> 第二天，松果稻草树来修取树叶穿过巢。它已经修好了，斯波特高兴极了。蒂米不再那么悲伤，他感谢莉莉的帮助。他们俩都笑了，直到回家的时间。这对家里来说是一个夜晚对天空更加精致而危险的一天。
> achieved tok/s: 47.076131



## Chinese Tinystory



[Chinese TinyStories](https://huggingface.co/datasets/52AI/TinyStoriesZh/tree/main) translated from English story data [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) , with a free [translator](https://pypi.org/project/deep-translator).

This is a translation example,  a chinese story is translated from one english story.

> Lily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n"Push me, Ben!" she says. Ben pushes her gently. Lily feels happy. She swings higher and higher. She laughs and shouts.\nBen watches Lily. He thinks she is cute. He wants to swing too. He waits for Lily to stop. But Lily does not stop. She swings faster and faster. She is having too much fun.\n"Can I swing too, Lily?" Ben asks. Lily does not hear him. She is too busy swinging. Ben feels sad. He walks away.\nLily swings so high that she loses her grip. She falls off the swing. She lands on the ground. She hurts her foot. She cries.\n"Ow, ow, ow!" she says. She looks for Ben. She wants him to help her. But Ben is not there. He is gone.\nLily feels sorry. She wishes she had shared the swing with Ben. She wishes he was there to hug her. She limps to the tree. She sees something hanging from a branch. It is Ben\'s hat. He left it for her.\nLily smiles. She thinks Ben is nice. She puts on his hat. She hopes he will come back. She wants to say sorry. She wants to be friends again.

> 莉莉和本是朋友。他们喜欢在公园里玩。有一天，他们看到一棵有秋千的大树。莉莉想尝试秋千。她跑到树旁，爬上秋千。\n“推我吧，本！”她说。本轻轻地推了她一下。莉莉感觉很幸福。她荡得越来越高。她又笑又叫。\n本看着莉莉。他觉得她很可爱。他也想摇摆。他等着莉莉停下来。但莉莉并没有停下来。她摆动得越来越快。她玩得太开心了。\n“我也可以荡秋千吗，莉莉？”本问。莉莉没有听见他的话。她正忙着荡秋千。本感到难过。他走开了。\n莉莉荡得太高，以至于她失去了抓力。她从秋千上摔下来。她降落在地上。她的脚受伤了。她哭了。\n“呜呜呜！”她说。她寻找本。她想要他帮助她。但本不在那儿。他已经去了。\n莉莉感到抱歉。她希望自己能和本一起荡秋千。她希望他能在那里拥抱她。她一瘸一拐地走向树。她看到树枝上挂着什么东西。这是本的帽子。他留给她了。\n莉莉微笑着。她认为本很好。她戴上他的帽子。她希望他能回来。她想说对不起。她想再次成为朋友。



## Traning

Before data precessing, you may check TOKENIZER_MODEL and TOKENIZER_BIN in your config(config.py). 

llama2enzh-52k(config.py)

```
TOKENIZER_MODEL = "tokenizers/llama2enzh/tokenizer.model" # the llama2.
TOKENIZER_BIN = "tokenizers/llama2enzh/tokenizer.bin" # binary version of the tokenizer for inference in C
```

baichuan-64k(config.py)

```
# https://huggingface.co/baichuan-inc/Baichuan-7B/blob/main/tokenizer.model
TOKENIZER_MODEL = "tokenizers/baichuan/tokenizer.model" # the baichuan sentencepiece tokenizer model
TOKENIZER_BIN = "tokenizers/baichuan/tokenizer.bin" # binary version of the tokenizer for inference in C
```

llama2en-32k(config.py)

```
TOKENIZER_MODEL = "tokenizers/llama2en/tokenizer.model" # the llama2-enzh.
TOKENIZER_BIN = "tokenizers/llama2en/tokenizer.bin" # binary version of the tokenizer for inference in C
```

When config is cheked,  we just need to run a single script to precess data.

```
# Download dataset. choice which dataset(en, enzh, zh) to use in config.py. 
>> python tinystories.py download
# tokenizer
>> python tinystories.py pretokenize
```

Now, you can train your model.

```
python train.py
```



### Custom Training 

You can refer to the resources to prepare a customized vocabulary, and then start training from scratch.

**new vocab**：

chinese-tokenizer example:  [train_tokenizer_from_scrath](https://github.com/chenyangMl/llama2.c-zh/blob/main/tokenizers/train_tokenizer_from_scrath.ipynb)

 HF-demo: https://huggingface.co/docs/tokenizers/pipeline

sentencepiece example: [sentencepiece_python_module_example](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb)

**extend vocab**：[sentencepiece add new vocab](https://github.com/google/sentencepiece/blob/9cf136582d9cce492ba5a0cfb775f9e777fe07ea/python/add_new_vocab.ipynb)

**reduce vocan**：[Reducing the SentencePiece Vocabulary Size of Pretrained NLP Models](https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/) |  [toknizer reduce](https://github.com/bojone/t5_in_bert4keras/blob/6cf50dbf3ffd3b4e9f36a59ee9f98356cf686de0/tokenizer/reduce.py)



### Validate  transformed model

After model conversion, verify the consistency of the output results of the Pytroch model and the C model. Note that the tokenzier. model in config.py needs to be aligned with the validation model.

```shell
>> python test_all.py [path_to_ckpt_dir]
eg: python test_all.py out/stories15M-llama2-enzh
```

> TOKENIZER_BIN=tokenizers/baichuan/tokenizer.bin
> ckpt_dir=out/demo2-enzh-baicuan0
> Loaded SentencePiece model from tokenizers/baichuan/tokenizer.model
> #words: 64000 - BOS ID: 1 - EOS ID: 2
> achieved tok/s: 43.202630
> c>>从前，有一个小女孩，名叫莉莉。她喜欢在阳光下外面玩耍。有一天，她和妈妈一起去公园。她看到一棵大树，树上挂着一个漂亮的球。
> p>>从前，有一个小女孩，名叫莉莉。她喜欢在阳光下外面玩耍。有一天，她和妈妈一起去公园。她看到一棵大树，树上挂着一个漂亮的球。
>
> ...
>
> it's good.



## Performance

| Model               | Tokens/S (Intel CPU @ 3.00GHz (server)) | Tokens/S (Intel CPU @ 1.7GHz (Mac Pro)) |
| ------------------- | :-------------------------------------: | :-------------------------------------: |
| Llama2-en(15M)      |                  69.14                  |                  58.82                  |
| Llama2-en(42M)      |                  23.58                  |                  21.57                  |
|                     |                                         |                                         |
| Llama2-enzh(15M+7M) |                  47.05                  |                  41.23                  |
|                     |                                         |                                         |
| baichuan(15M+8M)    |                   42.                   |                  38.94                  |
|                     |                                         |                                         |



## TODO List:

- Translate and organize the remaining training data.
- Translate and organize SFT training, fine-tuning SLM with  instruction trainset.
- The current Chinese English model is a demo model trained with 50w of Chinese data, with a poor effect.  Continue training with additional data.



## Reference

llama2.c: https://github.com/karpathy/llama2.c

Baichuan7B: https://huggingface.co/baichuan-inc/Baichuan-7B

Llama2-Chinese： https://github.com/FlagAlpha/Llama2-Chinese



## Relevant content

llama.cpp: https://github.com/ggerganov/llama.cpp    | [Georgi Gerganov](https://twitter.com/ggerganov)

whisper.cpp: https://github.com/ggerganov/whisper.cpp  | | [Georgi Gerganov](https://twitter.com/ggerganov)

Fastllm:  https://github.com/ztxz16/fastllm

vLLM:  https://github.com/vllm-project/vllm

## [License](https://github.com/karpathy/llama2.c)

MIT