## llama2.c-zh

中文 | [English](/assets/readme_en.md)

<p align="center">
  <img src="assets/llama2-zh.jpg" width="300" height="300" alt="Cute Llama-zh">
</p>

感谢 [Andrej Karpathy](https://github.com/karpathy)提供 [llama2.c](https://github.com/karpathy/llama2.c)工程，llama2.c 在llama2的基础上致力于研究小模型的能力边界和工程实践。

本工程希望将研究场景从英文拓展到中文。提供扩展的llama2词表和 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)的中文翻译数据TinyStory-zh，及训练的中英混合模型。




## 更新日志

**2028.08.12：**

-  1M中文TinyStory数据（资源有限仅翻译了[story, feature, word, summary]中story部分）

-   提供 [Llama2-52k-enzh](https://huggingface.co/52AI/tinyllamas_zh/tree/main)， [baichuan-64k-enzh](https://huggingface.co/52AI/tinyllamas_zh/tree/main)的中英混合模型，模型训练，c语言推理。

  

## 怎么用

```
git clone git clone https://github.com/chenyangMl/llama2.c-zh.git
cd llama2.c-zh
make run
```

参数运行llama2enzh：

```
wget https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-llama2-enzh.bin
./run stories15M-llama2-enzh.bin -k tokenizers/llama2enzh/tokenizer.bin -i "从前，有一个小女孩" -t 0.9
```

> 从前，有一个小女孩，名叫莉莉。她喜欢在海滩上玩耍。有一天，她看到一只海豚在水中瑟瑟发抖。 “你为什么斯纳格斯先生？”她问。 “我不知道，”小海豚说。 “我们去取暖吧！”
> 莉莉和斯纳格斯先生在那里保持安静。当他们到达时，他们看到海豚正在摩擦他的鼻子。莉莉和斯纳格斯先生停下来看看发生了什么事。海豚微笑着说：“我是一种长颈鹿生活在海洋中，你想摸你吗？”
> 莉莉和斯纳密斯先生抬头一看，发现海豚已经变得非常明亮和漂亮。 “哇！”莉莉说。 “它们真漂亮！” “我们可以摸一下吗？”她问。 “当然！”海豚说。 “但是要小心，提眼，很害怕。”莉莉和斯纳格斯先生深深地吸了一口气，看着海豚游走。他们既高兴又好奇。
> achieved tok/s: 41.989401

PS: 这里有可能会出英文，是正常的。



参数运行baichuan：

```
wget https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-baichuan.bin
./run stories15M-baichuan.bin -k tokenizers/baichuan/tokenizer.bin -i "One day, Lily met a Shoggoth" -t 0.8
```

> One day, Lily met a Shoggoth. She was very excited and asked her mom, "Can I go explore the woods?" Her mom replied, "Well, you can go explore as well. But be careful and don't go too far away."
> Rhera nodded eagerly and went outside. As she walked deeper into the woods, she noticed something strange - a strange shape floating in the air. She thought it was a spot that was growing and there was a big pond.
> Luckily, she figured out a plan. She spotted a sparkling fish swimming in the water and she was so excited that she ran back to her mom and said, "Mom, I found a special fish!" Her mom smiled and said, "That's a very creative fish, Amy. I've made it to find a special place in the woods."
> So, Peter and his mom went back home, and Amy was soon safe and sound filled with all the interesting creatures.
> achieved tok/s: 37.952023



## 拓展词表

如下示列，原始的llama2主要统计英文语料，llama2_enzh通过加入中文语料统计并合并原始的词表形成新的52k词表。使用原始的32K [llama2-en](https://github.com/karpathy/llama2.c/blob/master/tokenizer.model)词表编码s消耗的token数是57，扩展后的llama2_enzh词表编码s消耗token数19。

假设我们任务场景是中文，同样训练一个max_length=512token的序列 。编码一个中文字符消耗tokens, llam2-en=1.8, llama2_enzh=0.59。粗略估算下，512tokens实际编码的中文字符长度llama2-en=512/1.8=284, llama2_enzh=512/0.59=867。这就是拓展原始词表的原因。

**从头训练词表**：https://huggingface.co/docs/tokenizers/pipeline

**扩展词表**：[sentencepiece add new vocab](https://github.com/google/sentencepiece/blob/9cf136582d9cce492ba5a0cfb775f9e777fe07ea/python/add_new_vocab.ipynb)

**缩减词表**：[Reducing the SentencePiece Vocabulary Size of Pretrained NLP Models](https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/) | [toknizer reduce](https://github.com/bojone/t5_in_bert4keras/blob/6cf50dbf3ffd3b4e9f36a59ee9f98356cf686de0/tokenizer/reduce.py)

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
print(f"编码一个中文字符消耗tokens, llam2-en={len(llama2_en)/len(s):.2}, llama2_enzh={len(llama2_enzh)/len(s):.2}")
#-----------------------------------
llama2_en=57, llama2_enzh=19 baichuan-enzh=23
llama2_en=32000, llama2_enzh=55296       baichuan-enzh=64000
llama2_en [29871, 31594, 30658, 30214, 30417, 30287, 30502, 30446, 30647, 232, 176, 172, 30214, 30548, 232, 146, 174, 235, 145, 140, 235, 145, 140, 30267, 232, 168, 188, 31823, 233, 175, 165, 30505, 31066, 30806, 234, 145, 172, 235, 131, 144, 30214, 233, 175, 166, 235, 184, 146, 30630, 231, 187, 192, 30210, 30830, 233, 159, 184, 30267]
llama2_enzh [29871, 40870, 30214, 32952, 41677, 30214, 40148, 34595, 34595, 30267, 32008, 32123, 40729, 42754, 30214, 35186, 37973, 46892, 30267]
编码一个中文字符消耗tokens, llam2-en=1.8, llama2_enzh=0.59
```



## 模型推理
[LLama2-en-32K](tokenizers/llama2en/tokenizer.model)模型

| model | dim | n_layers | n_heads | max context length | batch size | train   set | parameters | val loss | download|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OG | 288 | 6 | 6 | 256 | 128 | TinyStory | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 1024 | / | TinyStory | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 1024 | / | TinyStory | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

[LLama2-enzh-55k](tokenizers/llama2enzh/tokenizer.model) 模型

| model | dim  | n_layers | n_heads | max context length | batch size | train set                       | parameters | val loss | download                                                     |
| ----- | ---- | -------- | ------- | ------------------ | ---------- | ------------------------------- | ---------- | -------- | ------------------------------------------------------------ |
| OG    | 288  | 6        | 6       | 256                | 96         | TinyStory +   TingStory-zh(50w) | xxx        | 2.14     | [stories15M-llama2-enzh.bin](https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-llama2-enzh.bin) |

[Baichuan-enzh-64k](tokenizers/baichuan/tokenizer.model) 模型

| model | dim  | n_layers | n_heads | max context length | batch size | train set                             | parameters | val loss | download                                                     |
| ----- | ---- | -------- | ------- | ------------------ | ---------- | ------------------------------------- | ---------- | -------- | ------------------------------------------------------------ |
| OG    | 288  | 6        | 6       | 256                | 64         | TinyStory +         TingStory-zh(50w) | xxx        | 1.92     | [stories15M-baichuan.bin](https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-baichuan.bin) |

相教于llama2-en, [LLama2-enzh-55k](tokenizers/llama2enzh/tokenizer.model) 和 [Baichuan-enzh-64k](tokenizers/baichuan/tokenizer.model) 模型的训练变化:  1) 词表修改  2）增加了中文数据(50w train, 10w eval)  3) 修改了 batch size . 其他均保持不变。



模型推理：

```
# 下载模型
wget https://huggingface.co/52AI/tinyllamas_zh/resolve/main/stories15M-baichuan.bin
./run stories15M-baichuan.bin -k tokenizers/baichuan/tokenizer.bin
```

> 从前，有一个名叫提米的女孩，她喜欢在森林里玩耍。她会跳进地上，吃饼干，堆着漂亮的树来筑巢。有一天，她看到地上有一个闪闪发亮的亮的香蕉。她想：“我可以用这个鸦做一颗爆姆花棒！”她妈妈来到她的面前，说道：“可以和日子分享分享吧，但记住，有时意外可能发生，对朋友来说可能会团队合作。”
> 莉莉想了想，然后说道：“我们可以从树上看松果香。这是个好主意！”他们回家并确保远离这些橄榄。
> 第二天，松果稻草树来修取树叶穿过巢。它已经修好了，斯波特高兴极了。蒂米不再那么悲伤，他感谢莉莉的帮助。他们俩都笑了，直到回家的时间。这对家里来说是一个夜晚对天空更加精致而危险的一天。
> achieved tok/s: 47.076131



## 训练数据

LM朝着越来越大的方向卷，而在小LM的方向，有研究者在探索小LM方向的边界能力，如想知道多小的语言模型仍然能流畅的说故事？

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 是在其做该方向时使用的一份关于小故事的场景数据。故事是由研究者使用GPT3.5, GPT4生成的，并且将故事难度限制在3~4岁小朋友能理解。 为了对齐，中文数据通过[翻译器](https://pypi.org/project/deep-translator/)将英文故事数据翻译而成。

> Lily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n"Push me, Ben!" she says. Ben pushes her gently. Lily feels happy. She swings higher and higher. She laughs and shouts.\nBen watches Lily. He thinks she is cute. He wants to swing too. He waits for Lily to stop. But Lily does not stop. She swings faster and faster. She is having too much fun.\n"Can I swing too, Lily?" Ben asks. Lily does not hear him. She is too busy swinging. Ben feels sad. He walks away.\nLily swings so high that she loses her grip. She falls off the swing. She lands on the ground. She hurts her foot. She cries.\n"Ow, ow, ow!" she says. She looks for Ben. She wants him to help her. But Ben is not there. He is gone.\nLily feels sorry. She wishes she had shared the swing with Ben. She wishes he was there to hug her. She limps to the tree. She sees something hanging from a branch. It is Ben\'s hat. He left it for her.\nLily smiles. She thinks Ben is nice. She puts on his hat. She hopes he will come back. She wants to say sorry. She wants to be friends again.

> 莉莉和本是朋友。他们喜欢在公园里玩。有一天，他们看到一棵有秋千的大树。莉莉想尝试秋千。她跑到树旁，爬上秋千。\n“推我吧，本！”她说。本轻轻地推了她一下。莉莉感觉很幸福。她荡得越来越高。她又笑又叫。\n本看着莉莉。他觉得她很可爱。他也想摇摆。他等着莉莉停下来。但莉莉并没有停下来。她摆动得越来越快。她玩得太开心了。\n“我也可以荡秋千吗，莉莉？”本问。莉莉没有听见他的话。她正忙着荡秋千。本感到难过。他走开了。\n莉莉荡得太高，以至于她失去了抓力。她从秋千上摔下来。她降落在地上。她的脚受伤了。她哭了。\n“呜呜呜！”她说。她寻找本。她想要他帮助她。但本不在那儿。他已经去了。\n莉莉感到抱歉。她希望自己能和本一起荡秋千。她希望他能在那里拥抱她。她一瘸一拐地走向树。她看到树枝上挂着什么东西。这是本的帽子。他留给她了。\n莉莉微笑着。她认为本很好。她戴上他的帽子。她希望他能回来。她想说对不起。她想再次成为朋友。



## 模型训练
数据预处理，预处理前请根据需要修改config.py中的tokenizer模型路径。

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

修改config.py中的LANGUAGE配置选择数据，运行数据加载和预处理脚本

```
#下载中英文数据
>> python tinystories.py download 
# tokenizer 数据
>> python tinystories.py pretokenize
```

运行训练脚本, 根据实际硬件情况修改下训练参数

```
python train.py
```



### 自定义训练

可参考下列方式准备自定义的词表，再从头开始训练.

**从头训练词表**：https://huggingface.co/docs/tokenizers/pipeline

**扩展词表**：[sentencepiece add new vocab](https://github.com/google/sentencepiece/blob/9cf136582d9cce492ba5a0cfb775f9e777fe07ea/python/add_new_vocab.ipynb)

**缩减词表**：[Reducing the SentencePiece Vocabulary Size of Pretrained NLP Models](https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/) |  [toknizer reduce](https://github.com/bojone/t5_in_bert4keras/blob/6cf50dbf3ffd3b4e9f36a59ee9f98356cf686de0/tokenizer/reduce.py)

或者使用更丰富的语料，进行模型训练。

https://github.com/crownpku/Awesome-Chinese-NLP

https://github.com/brightmart/nlp_chinese_corpus

https://github.com/fighting41love/funNLP

https://github.com/chenking2020/FindTheChatGPTer

https://github.com/esbatmop/MNBVC

https://github.com/shjwudp/shu/tree/master/books



### 模型转换验证

 模型转换后，验证pytroch模型和c模型输出结果的一致性。 注意config.py中的tokenzier.model需要和验证模型对齐。

```
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



## 性能测试

| Model               | Tokens/S (Intel CPU @ 3.00GHz (服务器)) | Tokens/S (Intel CPU @ 1.7GHz (Mac Pro)) |
| ------------------- | :-------------------------------------: | :-------------------------------------: |
| Llama2-en(15M)      |                  69.14                  |                  58.82                  |
| Llama2-en(42M)      |                  23.58                  |                  21.57                  |
|                     |                                         |                                         |
| Llama2-enzh(15M+7M) |                  47.05                  |                  41.23                  |
|                     |                                         |                                         |
| baichuan(15M+8M)    |                   42.                   |                  38.94                  |
|                     |                                         |                                         |

TODO List:

- 翻译整理剩余的pretraining训练数据。
- 翻译整理SFT训练，需要的指令微调中文数据。
- 当前中英模型是中文数据50w训练的demo模型, 效果不好，增加数据继续训练。



## 参考

llama2.c: https://github.com/karpathy/llama2.c

Baichuan7B: https://huggingface.co/baichuan-inc/Baichuan-7B

Llama2-Chinese： https://github.com/FlagAlpha/Llama2-Chinese



## 相关内容

llama.cpp(最早的c++推理LLM工程)  https://github.com/ggerganov/llama.cpp    | [Georgi Gerganov](https://twitter.com/ggerganov)

whisper.cpp(cpp推理ASR工程) https://github.com/ggerganov/whisper.cpp   | [Georgi Gerganov](https://twitter.com/ggerganov)

Fastllm(c++加速LLM推理): https://github.com/ztxz16/fastllm

vLLM(高效推理和部署) https://github.com/vllm-project/vllm

## [License](https://github.com/karpathy/llama2.c)

MIT