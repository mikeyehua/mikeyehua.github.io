# InstructLab

如今，大语言模型在很多场合蓬勃发展，我们都见证了人工智能给世界带来的重大影响，尤其是*ChatGPT*和其他大型语言模型的发布，我们都相信它将在未来几年改变我们的日常生活。但大多数模型仍是各自为政。虽然大语言模型 (LLM) 具有巨大的潜力，但它们也面临着挑战。使用 LLM 需要高质量的训练数据、专业技能和知识以及大量的计算资源。分叉和重新训练模型的过程也很耗时且成本高昂。开源社区通常为模型的生成贡献最多，但他们的贡献可能需要数月或数年才能合并回基础模型 （如果他们能回归的话） 


### 传统的RAG （检索增强生成）流程

传统方式没有社区，没有办法贡献代码，也没有办法丰富数据集

当前有许多项目正在采用开源的大语言模型，例如Llama、Mixtral，但它们遇到了三个主要挑战：

1. 直接对LLMs进行贡献不容易。重新训练新的模型会导致分叉，这让使用者不得不在不易扩展的模型之间做出选择。对于模型创建者来说，维护这些分叉的成本高且困难。
2. 对于贡献想法需要具备人工智能/机器学习专业知识的限制。一个人必须学习如何分叉、训练和优化模型才能实现自己的想法，这是一个高门槛的要求。
3. 缺乏有关分叉模型的社区治理或最佳实践的指导、审查和分发。

例如：*HuggingFace*上发布的许多*LLM*仅包含用于推理的成果 — — 它们周围没有社区，没有办法贡献代码，也没有办法丰富数据集。

所需的每个微调都**完全依赖于用户的责任**。


如果你仔细想想，你会发现它极大地减缓了创新。开源给软件行业带来的真正文化转变之一是能够**围绕**开源项目进行协作，从而创造出更好的解决方案和产品。公司和个人可以共享和贡献代码、修复错误、构建新功能，作为一个拥有共同目标的社区的一部分——**持续改进**。

人工智能的生命周期变得与传统软件非常相似——它是使用已知的编程语言和框架开发的，它可以被打包并构建到容器中，我们测试它，监控它，部署它。

![传统方式没有社区，没有办法贡献代码，也没有办法丰富数据集]

传统方式没有社区，没有办法贡献代码，也没有办法丰富数据集

那么我们如何利用已经做的事情来创建人工智能领域的社区呢？

我们如何合作、贡献知识和共享数据集以获得更好、更准确的人工智能模型？

[InstructLab](https://instructlab.ai/)

## **InstructLab 框架**

![InstructLab 社区模型将使用最新贡献进行更新，并定期在 Hugging Face 上分享。]

InstructLab 社区模型将使用最新贡献进行更新，并定期在 Hugging Face 上分享。

InstructLab 项目更像任何其他开源软件项目，提供了一种开源的生成式 AI 方法，它为社区提供了创建和合并 LLM 更改的工具，支持定期构建并且增强已有的预训练的大语言模型，而无需从头开始重新训练模型。这种方法不仅降低成本、消除测试和实验障碍，并且保证了一致性 — 即确保模型的答案准确、公正且符合其用户和创建者的目标。

InstructLab 的工作原理是利用 LLM 生成的高质量示例来增强人工整理的数据，从而降低数据创建成本。然后可以使用 InstructLab 生成的数据来定制或改进基础模型，而无需重新训练它，从而节省更多成本。IBM Research 已使用 InstructLab 生成合成数据，以改进其用于语言和[代码的开源](https://research.ibm.com/blog/granite-code-models-open-source)[Granite 模型](https://www.ibm.com/blog/building-ai-for-business-ibms-granite-foundation-models/)。

它提供**单一工具**来下载、提供、测试和训练*LLM*，以便任何人都可以贡献和改进现有功能——无论是内部还是外部，面向更广泛的社区。

InstructLab 使社区贡献者能够向特定模型**添加额外的“技能”或“知识”**

技能和知识的分类有助于识别所需能力的差距，然后在**合成数据中**生成足够的多样性以有效地调整基础模型。可以将InstructLab视为一个试验厨房，用于尝试和提交用于生成合成数据的新“配方”，以教授 LLM 新知识和技能。

**通过分类法**，[LAB](https://research.ibm.com/blog/LLM-generated-data)可以创建与您想要添加到模型中的任务相对应的高质量数据。分类法是迄今为止在 InstructLab 数据上调优的 LLM 所学到的知识的层次结构图，可轻松识别和填补漏洞。

InstructLab 的训练方案将新信息吸收到模型中，而不会导致模型覆盖之前学到的内容。基础模型在漫长的预训练阶段注入了核心知识和能力。如果需要进行实质性改进，则必须重新训练预先训练的基础模型。

**这个项目让不懂[transformer模型](https://www.zhihu.com/search?q=transformer%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%22699500296%22%7D), 不懂LLM的人, 也能训练自己的模型。**共同构建更好的模型，就像参与开源软件项目一样。

![**L**arge-scale **A**lignment for chat**B**ots]

**L**arge-scale **A**lignment for chat**B**ots

---

### **Install with Apple Metal on M1/M2/M3 Macs**

```bash
python3 -m venv --upgrade-deps venv
source venv/bin/activate
pip cache remove llama_cpp_python
pip install instructlab
```

```bash
/Users/yehua/instructlab

~/instructlab  source venv/bin/activate
(venv)  ~/instructlab  ilab chat
/Users/yehua/instructlab/venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
╭──────────────────────────────────────────────────────────────────────────────── system ────────────────────────────────────────────────────────────────────────────────╮
│ Welcome to InstructLab Chat w/ MODELS/MERLINITE-7B-LAB-Q4_K_M.GGUF (type /h for help)                                                                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>>                                                                                                                                                           [S][default]
```

### 下载 IBM Granite 模型

先准备好 HF Token = `xxxxxxxxxxxxxxxxxx`

https://huggingface.co/settings/tokens

如果需要，设置HF Token

```bash
HF_TOKEN=<YOUR HUGGINGFACE TOKEN GOES HERE> ilab download --repository=TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF --filename=mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
```

下载其他model

```bash
$ ilab download --repository instructlab/granite-7b-lab-GGUF --filename granite-7b-lab-Q4_K_M.gguf #IBM Granite
$ ilab download --repository QuantFactory/Qwen2-7B-Instruct-deccp-GGUF --filename Qwen2-7B-Instruct-deccp.Q4_K_M.gguf #通义千问
```

根据CPU核心，调整线程数

```bash
$ ilab serve --model-path models/granite-7b-lab-Q4_K_M.gguf --num-threads 14 --max-ctx-size 20480
$ ilab chat -m models/granite-7b-lab-Q4_K_M.gguf
```

### 添加 Knowledge 到 taxonomies

The LAB method is driven by [taxonomies](https://github.com/instructlab/taxonomy), which are largely created manually and with care.

`qna.yaml`

```yaml
version: 2
task_description: <string>
created_by: <string>
seed_examples: #需要提供至少两个相关的问答作为种子
  - question: <string>
    answer: |
      <multi-line string>
  - context: |
      <multi-line string>
    question: <string>
    answer: |
      <multi-line string>
  ...
```

**训练数据**

```yaml
version: 2
task_description: Teach the Large Language Model about the movie Oppenheimer
created_by: IBM Ecosystems Engineering SI Lab
domain: movie
seed_examples:
- answer: |
    The movie “Oppenheimer” was written, directed, and produced by Christopher Nolan1.
  question: Who directed the movie “Oppenheimer”?
- answer: |
    The movie follows the life of J. Robert Oppenheimer, the American theoretical physicist who helped develop the first nuclear weapons during World War II.
  question: What is the movie “Oppenheimer” about?
- answer: |
    The movie starred Cillian Murphy as Oppenheimer, alongside Robert Downey Jr. as the United States Atomic Energy Commission member Lewis Strauss. The ensemble supporting cast includes Emily Blunt, Matt Damon, Florence Pugh, Josh Hartnett, Casey Affleck, Rami Malek, and Kenneth Branagh
  question: Who starred in the movie “Oppenheimer”?
- answer: |
    The movie “Oppenheimer” was released on July 21, 2023
  question: When was the movie “Oppenheimer” released?
- answer: |
    The movie “Oppenheimer” received critical acclaim and won seven Academy Awards, including Best Picture, Best Director for Nolan, Best Actor for Murphy and Best Supporting Actor for Downey. It grossed over $976 million worldwide, becoming the third-highest-grossing film of 2023, the highest-grossing World War II-related film, the highest-grossing biographical film and the second-highest-grossing R-rated film
  question: How was the movie “Oppenheimer” received?
document:
  repo: https://github.com/mikeyehua/instructlab.git
  commit: 76a2773
  patterns:
  - Oppenheimer.md
```

`把上面的 问答文件 qna.yaml和 元数据文件 attribution.txt放在 taxonomy/knowledge/movies/oppenheimer 目录下`

原始参考数据来源`document`放在这里：

[instructlab/Oppenheimer.md at main · mikeyehua/instructlab](https://github.com/mikeyehua/instructlab/blob/main/Oppenheimer.md)

原文参考：https://github.com/syedaameena/InstructLab-Skill/blob/main/QnA.yaml?utm_source=ibm_developer&utm_content=in_content_link&utm_id=tutorials_awb-tuning-pretrained-llms-instructlab

- `document`: The source of your knowledge contribution.
    - `repo`: The URL to your repository that holds your knowledge markdown files.
    - `commit`: The SHA of the commit in your repository with your knowledge markdown files.
    - `patterns`: A list of glob patterns specifying the markdown files in your repository. Any glob pattern that starts with ``, such as `.md`, must be quoted due to YAML rules. For example, `"*.md"`.

<aside>
💡 YAML 测试站：

https://www.yamllint.com/

</aside>

**Knowledge: YAML examples**

<aside>
💡 **在没有生成新的训练数据集之前，用下面的问题测试，返回出的答案，可以看出来信息不正确，不能达到预期**

> Who starred in the movie Oppenheimer?
> 

elapsed 9.513 seconds -
The movie "Oppenheimer" is **an upcoming film** adaptation of the Pulitzer Prize-winning biography "American Prometheus: The Life of J. Robert Oppenheimer" by Kira Davis. Cillian Murphy has been cast as the title character, J. Robert Oppenheimer. Other notable stars who have joined the cast include Matt Damon as Leslie Groves, the head of the Yale project and the father of Oppenheimer's wife, Emily Blunt as Katherine Manning Oppenheimer, the wife of J. Robert Oppenheimer, and Alden Ehrenreich as Donald Oppenheimer, their son.

The film is directed by Christopher Nolan, known for his work on blockbuster movies like "Batman Begins," "The Dark Knight," and "Interstellar." "Oppenheimer" is set to explore the life of J. Robert Oppenheimer, the man who led the project that developed the first atomic bomb during World War II. The movie is expected to delve into the moral complexities of nuclear power and the human cost of scientific progress.

While there isn't much information available about the plot yet, I can provide you with some context on the historical figures and events that "Oppenheimer" might depict. The film will likely cover Oppenheimer's early life, his education at the prestigious Institute for Advanced Studies in Princeton, New Jersey, and his work at the Los Alamos National Laboratory during the Manhattan Project. Additionally, the movie may explore Oppenheimer's struggles with the moral implications of nuclear weapons and his eventual condemnation by the U.S. government during the Red Scare.

**Stay tuned for more updates** on "Oppenheimer" as filming progresses and **more information becomes available**. If you have any questions about the cast, directors, or historical context of the movie, please let me know!

> What are the release dates for Oppenheimer movie?
> 

elapsed 5.935 seconds -
As of my last update, the release date for "Oppenheimer" **is not yet set**. The film was announced in 2021, and production has recently started. Christopher Nolan, the director, often takes his time to perfect his films before releasing them. Given the complexity of the subject matter and the intricate plot, it's likely that "Oppenheimer" will require a considerable amount of post-production work.

**Once filming is complete and the movie is ready for release**, the production team will announce an official date. Typically, this happens several months before the release date. I recommend checking official film news sources and the film's official website for the most up-to-date information on the release date.

In the meantime, you can look forward to other movies released by Christopher Nolan or explore other fascinating historical figures and events that have been brought to the big screen. For example, "Dunkirk" (2017) and "Interstellar" (2014) are two of Nolan's previous films that showcase his ability to create immersive and thought-provoking cinematic experiences.

</aside>

### 验证数据

`ilab diff` 通过运行命令列出新数据并确保其在分类路径中正确注册，从而列出并验证新的数据。

### 生成合成数据集

通过运行 `ilab generate` 命令来根据我们输入的问答生成更广泛的数据集，该命令根据 taxonomies 存储库中新添加的  knowledge 生成合成数据集。

```yaml
ilab generate --model models/granite-7b-lab-Q4_K_M.gguf --num-instructions 100 --num-cpus 20 --server-ctx-size 20480
```

可以看到生成合并数据的时候，系统调用了Apple Silicon的GPU加速

您可以在输出中看到生成的新合成数据集。如果您对生成的数据集不满意，可以按 退出该过程`ctrl + c`。修改文件中的示例`qna.yaml`，然后重新运行`generate`命令。

此过程将需要一些时间，具体取决于您的系统。在我的 M1 Mac Pro 上大约需要 21 分钟。您可以在输出中看到 ETA。


生成合成数据后，您将看到已生成样本数和已丢弃样本数的摘要

```yaml
INFO 2024-06-20 15:37:52,612 generate_data.py:608 101 instructions generated, 10 discarded due to format (see generated/discarded_granite-7b-lab-Q4_K_M_2024-06-20T15_16_44.log), 2 discarded due to rouge score
INFO 2024-06-20 15:37:52,612 generate_data.py:612 Generation took **1270.44s

-rw-r--r--  1 yehua  staff     9242  6 20 15:36 discarded_granite-7b-lab-Q4_K_M_2024-06-20T15_16_44.log #丢弃的数据集（日志文件）
-rw-r--r--  1 yehua  staff  6280617  6 20 15:37 generated_granite-7b-lab-Q4_K_M_2024-06-20T15_16_44.json #生成的数据集（json文件）
-rw-r--r--  1 yehua  staff     2511  6 20 15:37 test_granite-7b-lab-Q4_K_M_2024-06-20T15_16_44.jsonl #测试数据集（jsonl文件）
-rw-r--r--  1 yehua  staff    94763  6 20 15:37 train_granite-7b-lab-Q4_K_M_2024-06-20T15_16_44.jsonl #训练数据集（jsonl文件）**
```

### 本地训练模型

一旦合成数据准备好了，您所要做的就是在终端中运行以下命令来训练模型：

为了训练该模式，我们将使用`ilab`指向本地`GGUF`文件的 CLI — 训练用于`cuda-toolkit`与底层 NVIDIA GPU 进行交互。

***注意***- 确保停止服务模型以便为训练阶段释放一些资源

`ilab train --gguf-model-path models/granite-7b-lab-Q4_K_M.gguf`

`ilab train --gguf-model-path models/granite-7b-lab-Q4_K_M.gguf --device 'cuda'`

```yaml
ilab train --gguf-model-path models/granite-7b-lab-Q4_K_M.gguf
/Users/yehua/instructlab/venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
[INFO] Loading
model-00003-of-00003.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.54G/4.54G [05:06<00:00, 8.66MB/s]
model-00001-of-00003.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.94G/4.94G [05:09<00:00, 8.71MB/s]
model-00002-of-00003.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.00G/5.00G [05:18<00:00, 9.35MB/s]
Fetching 11 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [05:20<00:00, 29.18s/it]
/Users/yehua/instructlab/venv/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.███████████████████████████████████████████████████████████████████████████████| 5.00G/5.00G [05:18<00:00, 25.9MB/s]
  warnings.warn(
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 644/644 [00:00<00:00, 628kB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.33k/2.33k [00:00<00:00, 1.47MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 668kB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 2.10MB/s]
added_tokens.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 119/119 [00:00<00:00, 67.5kB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [00:00<00:00, 207kB/s]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
dtype=mlx.core.float16
[INFO] Quantizing
Using model_type='mistral'
Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Total parameters 1244.079M
Trainable parameters 1.704M
Loading datasets
Training
Epoch 1: Iter 1: Val loss 2.087, Val took 26.183s
Iter 010: Train loss 1.848, It/sec 0.167, Tokens/sec 132.733
Epoch 1: Iter 10: Val loss 1.232, Val took 25.780s
Iter 10: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-010.npz.
Iter 020: Train loss 1.253, It/sec 0.195, Tokens/sec 133.049
Epoch 1: Iter 20: Val loss 1.053, Val took 25.881s
Iter 20: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-020.npz.
Iter 030: Train loss 0.924, It/sec 0.156, Tokens/sec 116.944
Epoch 2: Iter 30: Val loss 0.977, Val took 26.585s
Iter 30: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-030.npz.
Iter 040: Train loss 0.932, It/sec 0.167, Tokens/sec 121.478
Epoch 2: Iter 40: Val loss 0.937, Val took 26.198s
Iter 40: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-040.npz.
Iter 050: Train loss 0.763, It/sec 0.158, Tokens/sec 117.815
Epoch 3: Iter 50: Val loss 0.924, Val took 26.306s
Iter 50: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-050.npz.
Iter 060: Train loss 0.787, It/sec 0.165, Tokens/sec 119.941
Epoch 3: Iter 60: Val loss 0.903, Val took 27.079s
Iter 60: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-060.npz.
Iter 070: Train loss 0.668, It/sec 0.158, Tokens/sec 117.039
Epoch 4: Iter 70: Val loss 0.927, Val took 26.463s
Iter 70: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-070.npz.
Iter 080: Train loss 0.655, It/sec 0.132, Tokens/sec 96.649
Epoch 4: Iter 80: Val loss 0.911, Val took 26.960s
Iter 80: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-080.npz.
Iter 090: Train loss 0.599, It/sec 0.149, Tokens/sec 110.820
Epoch 5: Iter 90: Val loss 0.967, Val took 26.356s
Iter 90: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-090.npz.
Iter 100: Train loss 0.542, It/sec 0.152, Tokens/sec 111.259
Epoch 5: Iter 100: Val loss 0.947, Val took 28.326s
Iter 100: Saved adapter weights to instructlab-merlinite-7b-lab-mlx-q/adapters-100.npz.
```

此过程将需要一些时间，具体取决于您的系统 配置和迭代次数。在我的 M1 MacBook Pro 上完成 100 次迭代大约需要 30 分钟。您可以在输出中看到 ETA。

目录中将创建一个新目录`ilab`，其名称类似于：`instructlab-merlinite-7b-lab`。此目录将包含新的模型权重和适配器。

```yaml
drwxr-xr-x  14 yehua  staff   448  6 20 18:20 instructlab-merlinite-7b-lab
drwxr-xr-x  20 yehua  staff   640  6 20 18:36 instructlab-merlinite-7b-lab-mlx-q
```

### 测试模型

通过运行命令来测试新训练的模型`ilab test`，以测试模型并验证其性能。

`格式： ilab test --data-dir my-data --model-dir models/ibm/merlinite-7b`

`ilab test --data-dir ./taxonomy_data --model-dir instructlab-merlinite-7b-lab-mlx-q`

```yaml
ilab test

system prompt: You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
[1]
 user prompt: Who directed the movie “Oppenheimer”?
expected output: The movie “Oppenheimer” was written, directed, and produced by Christopher Nolan1.

-----model output BEFORE training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
LoRA init skipped
Total parameters 1242.375M
Trainable parameters 0.000M
Loading datasets
LoRA loading skipped
Generating
==========
Christopher Nolan
==========

-----model output AFTER training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Total parameters 1244.079M
Trainable parameters 1.704M
Loading datasets
Generating
==========
Christopher Nolan directed the movie “Oppenheimer.” He is known for his involvement in the production of the film and was chosen by the studio to take on this project.
==========
[2]
 user prompt: What is the movie “Oppenheimer” about?
expected output: The movie follows the life of J. Robert Oppenheimer, the American theoretical physicist who helped develop the first nuclear weapons during World War II.

-----model output BEFORE training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
LoRA init skipped
Total parameters 1242.375M
Trainable parameters 0.000M
Loading datasets
LoRA loading skipped
Generating
==========
"Oppenheimer" is a film that delves into the life and times of J. Robert Oppenheimer, the brilliant physicist who led the Manhattan Project during World War II. The movie offers a gripping and thought-provoking portrayal of Oppenheimer's personal and professional journey while working on the development of the atomic bomb. Here are some key aspects of the movie and the historical context that you might find useful for your article:

1.
==========

-----model output AFTER training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Total parameters 1244.079M
Trainable parameters 1.704M
Loading datasets
Generating
==========
"Oppenheimer" is a biographical film directed by Christopher Nolan that tells the story of J. Robert Oppenheimer, the father of the atomic bomb and the director of the Los Alamos Laboratory during World War II. The movie explores Oppenheimer's personal life, his contributions to the development of nuclear weapons, and the impact of his actions on the world.

The film portrays Oppenheimer's guilt over the destruction caused by the
==========
[3]
 user prompt: Who starred in the movie “Oppenheimer”?
expected output: The movie starred Cillian Murphy as Oppenheimer, alongside Robert Downey Jr. as the United States Atomic Energy Commission member Lewis Strauss. The ensemble supporting cast includes Emily Blunt, Matt Damon, Florence Pugh, Josh Hartnett, Casey Affleck, Rami Malek, and Kenneth Branagh

-----model output BEFORE training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
LoRA init skipped
Total parameters 1242.375M
Trainable parameters 0.000M
Loading datasets
LoRA loading skipped
Generating
==========
"Oppenheimer" is a historical drama film directed by Christopher Nolan, with an impressive cast and crew. The lead role of Dr. J. Robert Oppenheimer, the physicist who oversaw the Manhattan Project, is portrayed by a well-known actor, Cillian Murphy. Additionally, the film features other esteemed actors, including Emily Blunt as Katherine, Olivia Thirlby as Leona Woods, Matthew Goode as Klaus Fuch
==========

-----model output AFTER training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Total parameters 1244.079M
Trainable parameters 1.704M
Loading datasets
Generating
==========
Cillian Murphy, Matt Damon, Emily Blunt, Robert Downey Jr., Gary Oldman, and Benny Safdie.
==========
[4]
 user prompt: When was the movie “Oppenheimer” released?
expected output: The movie “Oppenheimer” was released on July 21, 2023

-----model output BEFORE training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
LoRA init skipped
Total parameters 1242.375M
Trainable parameters 0.000M
Loading datasets
LoRA loading skipped
Generating
==========
I'm sorry for the confusion, but I couldn't find information about the release date of the movie "Oppenheimer." Would you like me to search for more details about the movie or answer a different question?
==========

-----model output AFTER training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Total parameters 1244.079M
Trainable parameters 1.704M
Loading datasets
Generating
==========
The movie was released in theaters on July 21, 2023.
==========
[5]
 user prompt: How was the movie “Oppenheimer” received?
expected output: The movie “Oppenheimer” received critical acclaim and won seven Academy Awards, including Best Picture, Best Director for Nolan, Best Actor for Murphy and Best Supporting Actor for Downey. It grossed over $976 million worldwide, becoming the third-highest-grossing film of 2023, the highest-grossing World War II-related film, the highest-grossing biographical film and the second-highest-grossing R-rated film

-----model output BEFORE training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
LoRA init skipped
Total parameters 1242.375M
Trainable parameters 0.000M
Loading datasets
LoRA loading skipped
Generating
==========
The movie "Oppenheimer" has received mixed reviews from critics and audiences alike. While some laud the film for its powerful performances and thought-provoking themes, others have criticized it for its pacing and structure.

Critics have praised the movie's direction, cinematography, and visual effects. The performances of the lead actors, particularly Cillian Murphy as J. Robert Oppenheimer, have been widely praised. However, some have criticized
==========

-----model output AFTER training----:

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Total parameters 1244.079M
Trainable parameters 1.704M
Loading datasets
Generating
==========
The movie "Oppenheimer" received mixed reviews from critics and audiences. It was praised for its visual effects and historical accuracy, but criticized for its length and slow pacing. Some viewers felt that the film could have been more engaging and emotionally resonant.
```

### **量化精细调整的模型**

现在训练已经完成，您应该期望`GGUF`在该模型路径下拥有新模型（在我们的例子中是模型目录）。

为了在未来的文章中使用该模型，我们需要它具有合理的大小，接下来，我们将量化模型

`ilab convert`

```bash
ilab convert

Loading pretrained model
Using model_type='mistral'
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 2024-06-20 18:48:22,055 lab.py:1343 deleting instructlab-merlinite-7b-lab-mlx-q...
[INFO] Loading
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
dtype=<class 'numpy.float16'>
INFO 2024-06-20 18:50:26,422 lab.py:1352 deleting instructlab-merlinite-7b-lab-mlx-q-fused...
Loading model file instructlab-merlinite-7b-lab-trained/model.safetensors
params = Params(n_vocab=32008, n_embd=4096, n_layer=32, n_ctx=32768, n_ff=14336, n_head=32, n_head_kv=8, n_experts=None, n_experts_used=None, f_norm_eps=1e-05, rope_scaling_type=None, f_rope_freq_base=10000.0, f_rope_scale=None, n_orig_ctx=None, rope_finetuned=None, ftype=None, path_model=PosixPath('instructlab-merlinite-7b-lab-trained'))
Found vocab files: {'spm': PosixPath('instructlab-merlinite-7b-lab-trained/tokenizer.model'), 'bpe': None, 'hfft': PosixPath('instructlab-merlinite-7b-lab-trained/tokenizer.json')}
Loading vocab file PosixPath('instructlab-merlinite-7b-lab-trained/tokenizer.model'), type 'spm'
Vocab info: <SentencePieceVocab with 32000 base tokens and 5 added tokens>
Special vocab info: <SpecialVocab with 0 merges, special tokens {'bos': 1, 'eos': 32000, 'unk': 0, 'pad': 32001}, add special tokens {'bos': False, 'eos': False}>
Permuting layer 0
Permuting layer 1
Permuting layer 2
Permuting layer 3
Permuting layer 4
Permuting layer 5
Permuting layer 6
Permuting layer 7
Permuting layer 8
Permuting layer 9
Permuting layer 10
Permuting layer 11
Permuting layer 12
Permuting layer 13
Permuting layer 14
Permuting layer 15
Permuting layer 16
Permuting layer 17
Permuting layer 18
Permuting layer 19
Permuting layer 20
Permuting layer 21
Permuting layer 22
Permuting layer 23
Permuting layer 24
Permuting layer 25
Permuting layer 26
Permuting layer 27
Permuting layer 28
Permuting layer 29
Permuting layer 30
Permuting layer 31
lm_head.weight                                   -> output.weight                            | F16    | [32008, 4096]
model.embed_tokens.weight                        -> token_embd.weight                        | F16    | [32008, 4096]
model.layers.0.input_layernorm.weight            -> blk.0.attn_norm.weight                   | F16    | [4096]
model.layers.0.mlp.down_proj.weight              -> blk.0.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.0.mlp.gate_proj.weight              -> blk.0.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.0.mlp.up_proj.weight                -> blk.0.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.0.post_attention_layernorm.weight   -> blk.0.ffn_norm.weight                    | F16    | [4096]
model.layers.0.self_attn.k_proj.weight           -> blk.0.attn_k.weight                      | F16    | [1024, 4096]
model.layers.0.self_attn.o_proj.weight           -> blk.0.attn_output.weight                 | F16    | [4096, 4096]
model.layers.0.self_attn.q_proj.weight           -> blk.0.attn_q.weight                      | F16    | [4096, 4096]
model.layers.0.self_attn.v_proj.weight           -> blk.0.attn_v.weight                      | F16    | [1024, 4096]
model.layers.1.input_layernorm.weight            -> blk.1.attn_norm.weight                   | F16    | [4096]
model.layers.1.mlp.down_proj.weight              -> blk.1.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.1.mlp.gate_proj.weight              -> blk.1.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.1.mlp.up_proj.weight                -> blk.1.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.1.post_attention_layernorm.weight   -> blk.1.ffn_norm.weight                    | F16    | [4096]
model.layers.1.self_attn.k_proj.weight           -> blk.1.attn_k.weight                      | F16    | [1024, 4096]
model.layers.1.self_attn.o_proj.weight           -> blk.1.attn_output.weight                 | F16    | [4096, 4096]
model.layers.1.self_attn.q_proj.weight           -> blk.1.attn_q.weight                      | F16    | [4096, 4096]
model.layers.1.self_attn.v_proj.weight           -> blk.1.attn_v.weight                      | F16    | [1024, 4096]
model.layers.10.input_layernorm.weight           -> blk.10.attn_norm.weight                  | F16    | [4096]
model.layers.10.mlp.down_proj.weight             -> blk.10.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.10.mlp.gate_proj.weight             -> blk.10.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.10.mlp.up_proj.weight               -> blk.10.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.10.post_attention_layernorm.weight  -> blk.10.ffn_norm.weight                   | F16    | [4096]
model.layers.10.self_attn.k_proj.weight          -> blk.10.attn_k.weight                     | F16    | [1024, 4096]
model.layers.10.self_attn.o_proj.weight          -> blk.10.attn_output.weight                | F16    | [4096, 4096]
model.layers.10.self_attn.q_proj.weight          -> blk.10.attn_q.weight                     | F16    | [4096, 4096]
model.layers.10.self_attn.v_proj.weight          -> blk.10.attn_v.weight                     | F16    | [1024, 4096]
model.layers.11.input_layernorm.weight           -> blk.11.attn_norm.weight                  | F16    | [4096]
model.layers.11.mlp.down_proj.weight             -> blk.11.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.11.mlp.gate_proj.weight             -> blk.11.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.11.mlp.up_proj.weight               -> blk.11.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.11.post_attention_layernorm.weight  -> blk.11.ffn_norm.weight                   | F16    | [4096]
model.layers.11.self_attn.k_proj.weight          -> blk.11.attn_k.weight                     | F16    | [1024, 4096]
model.layers.11.self_attn.o_proj.weight          -> blk.11.attn_output.weight                | F16    | [4096, 4096]
model.layers.11.self_attn.q_proj.weight          -> blk.11.attn_q.weight                     | F16    | [4096, 4096]
model.layers.11.self_attn.v_proj.weight          -> blk.11.attn_v.weight                     | F16    | [1024, 4096]
model.layers.12.input_layernorm.weight           -> blk.12.attn_norm.weight                  | F16    | [4096]
model.layers.12.mlp.down_proj.weight             -> blk.12.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.12.mlp.gate_proj.weight             -> blk.12.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.12.mlp.up_proj.weight               -> blk.12.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.12.post_attention_layernorm.weight  -> blk.12.ffn_norm.weight                   | F16    | [4096]
model.layers.12.self_attn.k_proj.weight          -> blk.12.attn_k.weight                     | F16    | [1024, 4096]
model.layers.12.self_attn.o_proj.weight          -> blk.12.attn_output.weight                | F16    | [4096, 4096]
model.layers.12.self_attn.q_proj.weight          -> blk.12.attn_q.weight                     | F16    | [4096, 4096]
model.layers.12.self_attn.v_proj.weight          -> blk.12.attn_v.weight                     | F16    | [1024, 4096]
model.layers.13.input_layernorm.weight           -> blk.13.attn_norm.weight                  | F16    | [4096]
model.layers.13.mlp.down_proj.weight             -> blk.13.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.13.mlp.gate_proj.weight             -> blk.13.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.13.mlp.up_proj.weight               -> blk.13.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.13.post_attention_layernorm.weight  -> blk.13.ffn_norm.weight                   | F16    | [4096]
model.layers.13.self_attn.k_proj.weight          -> blk.13.attn_k.weight                     | F16    | [1024, 4096]
model.layers.13.self_attn.o_proj.weight          -> blk.13.attn_output.weight                | F16    | [4096, 4096]
model.layers.13.self_attn.q_proj.weight          -> blk.13.attn_q.weight                     | F16    | [4096, 4096]
model.layers.13.self_attn.v_proj.weight          -> blk.13.attn_v.weight                     | F16    | [1024, 4096]
model.layers.14.input_layernorm.weight           -> blk.14.attn_norm.weight                  | F16    | [4096]
model.layers.14.mlp.down_proj.weight             -> blk.14.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.14.mlp.gate_proj.weight             -> blk.14.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.14.mlp.up_proj.weight               -> blk.14.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.14.post_attention_layernorm.weight  -> blk.14.ffn_norm.weight                   | F16    | [4096]
model.layers.14.self_attn.k_proj.weight          -> blk.14.attn_k.weight                     | F16    | [1024, 4096]
model.layers.14.self_attn.o_proj.weight          -> blk.14.attn_output.weight                | F16    | [4096, 4096]
model.layers.14.self_attn.q_proj.weight          -> blk.14.attn_q.weight                     | F16    | [4096, 4096]
model.layers.14.self_attn.v_proj.weight          -> blk.14.attn_v.weight                     | F16    | [1024, 4096]
model.layers.15.input_layernorm.weight           -> blk.15.attn_norm.weight                  | F16    | [4096]
model.layers.15.mlp.down_proj.weight             -> blk.15.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.15.mlp.gate_proj.weight             -> blk.15.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.15.mlp.up_proj.weight               -> blk.15.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.15.post_attention_layernorm.weight  -> blk.15.ffn_norm.weight                   | F16    | [4096]
model.layers.15.self_attn.k_proj.weight          -> blk.15.attn_k.weight                     | F16    | [1024, 4096]
model.layers.15.self_attn.o_proj.weight          -> blk.15.attn_output.weight                | F16    | [4096, 4096]
model.layers.15.self_attn.q_proj.weight          -> blk.15.attn_q.weight                     | F16    | [4096, 4096]
model.layers.15.self_attn.v_proj.weight          -> blk.15.attn_v.weight                     | F16    | [1024, 4096]
model.layers.16.input_layernorm.weight           -> blk.16.attn_norm.weight                  | F16    | [4096]
model.layers.16.mlp.down_proj.weight             -> blk.16.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.16.mlp.gate_proj.weight             -> blk.16.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.16.mlp.up_proj.weight               -> blk.16.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.16.post_attention_layernorm.weight  -> blk.16.ffn_norm.weight                   | F16    | [4096]
model.layers.16.self_attn.k_proj.weight          -> blk.16.attn_k.weight                     | F16    | [1024, 4096]
model.layers.16.self_attn.o_proj.weight          -> blk.16.attn_output.weight                | F16    | [4096, 4096]
model.layers.16.self_attn.q_proj.weight          -> blk.16.attn_q.weight                     | F16    | [4096, 4096]
model.layers.16.self_attn.v_proj.weight          -> blk.16.attn_v.weight                     | F16    | [1024, 4096]
model.layers.17.input_layernorm.weight           -> blk.17.attn_norm.weight                  | F16    | [4096]
model.layers.17.mlp.down_proj.weight             -> blk.17.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.17.mlp.gate_proj.weight             -> blk.17.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.17.mlp.up_proj.weight               -> blk.17.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.17.post_attention_layernorm.weight  -> blk.17.ffn_norm.weight                   | F16    | [4096]
model.layers.17.self_attn.k_proj.weight          -> blk.17.attn_k.weight                     | F16    | [1024, 4096]
model.layers.17.self_attn.o_proj.weight          -> blk.17.attn_output.weight                | F16    | [4096, 4096]
model.layers.17.self_attn.q_proj.weight          -> blk.17.attn_q.weight                     | F16    | [4096, 4096]
model.layers.17.self_attn.v_proj.weight          -> blk.17.attn_v.weight                     | F16    | [1024, 4096]
model.layers.18.input_layernorm.weight           -> blk.18.attn_norm.weight                  | F16    | [4096]
model.layers.18.mlp.down_proj.weight             -> blk.18.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.18.mlp.gate_proj.weight             -> blk.18.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.18.mlp.up_proj.weight               -> blk.18.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.18.post_attention_layernorm.weight  -> blk.18.ffn_norm.weight                   | F16    | [4096]
model.layers.18.self_attn.k_proj.weight          -> blk.18.attn_k.weight                     | F16    | [1024, 4096]
model.layers.18.self_attn.o_proj.weight          -> blk.18.attn_output.weight                | F16    | [4096, 4096]
model.layers.18.self_attn.q_proj.weight          -> blk.18.attn_q.weight                     | F16    | [4096, 4096]
model.layers.18.self_attn.v_proj.weight          -> blk.18.attn_v.weight                     | F16    | [1024, 4096]
model.layers.19.input_layernorm.weight           -> blk.19.attn_norm.weight                  | F16    | [4096]
model.layers.19.mlp.down_proj.weight             -> blk.19.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.19.mlp.gate_proj.weight             -> blk.19.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.19.mlp.up_proj.weight               -> blk.19.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.19.post_attention_layernorm.weight  -> blk.19.ffn_norm.weight                   | F16    | [4096]
model.layers.19.self_attn.k_proj.weight          -> blk.19.attn_k.weight                     | F16    | [1024, 4096]
model.layers.19.self_attn.o_proj.weight          -> blk.19.attn_output.weight                | F16    | [4096, 4096]
model.layers.19.self_attn.q_proj.weight          -> blk.19.attn_q.weight                     | F16    | [4096, 4096]
model.layers.19.self_attn.v_proj.weight          -> blk.19.attn_v.weight                     | F16    | [1024, 4096]
model.layers.2.input_layernorm.weight            -> blk.2.attn_norm.weight                   | F16    | [4096]
model.layers.2.mlp.down_proj.weight              -> blk.2.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.2.mlp.gate_proj.weight              -> blk.2.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.2.mlp.up_proj.weight                -> blk.2.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.2.post_attention_layernorm.weight   -> blk.2.ffn_norm.weight                    | F16    | [4096]
model.layers.2.self_attn.k_proj.weight           -> blk.2.attn_k.weight                      | F16    | [1024, 4096]
model.layers.2.self_attn.o_proj.weight           -> blk.2.attn_output.weight                 | F16    | [4096, 4096]
model.layers.2.self_attn.q_proj.weight           -> blk.2.attn_q.weight                      | F16    | [4096, 4096]
model.layers.2.self_attn.v_proj.weight           -> blk.2.attn_v.weight                      | F16    | [1024, 4096]
model.layers.20.input_layernorm.weight           -> blk.20.attn_norm.weight                  | F16    | [4096]
model.layers.20.mlp.down_proj.weight             -> blk.20.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.20.mlp.gate_proj.weight             -> blk.20.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.20.mlp.up_proj.weight               -> blk.20.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.20.post_attention_layernorm.weight  -> blk.20.ffn_norm.weight                   | F16    | [4096]
model.layers.20.self_attn.k_proj.weight          -> blk.20.attn_k.weight                     | F16    | [1024, 4096]
model.layers.20.self_attn.o_proj.weight          -> blk.20.attn_output.weight                | F16    | [4096, 4096]
model.layers.20.self_attn.q_proj.weight          -> blk.20.attn_q.weight                     | F16    | [4096, 4096]
model.layers.20.self_attn.v_proj.weight          -> blk.20.attn_v.weight                     | F16    | [1024, 4096]
model.layers.21.input_layernorm.weight           -> blk.21.attn_norm.weight                  | F16    | [4096]
model.layers.21.mlp.down_proj.weight             -> blk.21.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.21.mlp.gate_proj.weight             -> blk.21.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.21.mlp.up_proj.weight               -> blk.21.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.21.post_attention_layernorm.weight  -> blk.21.ffn_norm.weight                   | F16    | [4096]
model.layers.21.self_attn.k_proj.weight          -> blk.21.attn_k.weight                     | F16    | [1024, 4096]
model.layers.21.self_attn.o_proj.weight          -> blk.21.attn_output.weight                | F16    | [4096, 4096]
model.layers.21.self_attn.q_proj.weight          -> blk.21.attn_q.weight                     | F16    | [4096, 4096]
model.layers.21.self_attn.v_proj.weight          -> blk.21.attn_v.weight                     | F16    | [1024, 4096]
model.layers.22.input_layernorm.weight           -> blk.22.attn_norm.weight                  | F16    | [4096]
model.layers.22.mlp.down_proj.weight             -> blk.22.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.22.mlp.gate_proj.weight             -> blk.22.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.22.mlp.up_proj.weight               -> blk.22.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.22.post_attention_layernorm.weight  -> blk.22.ffn_norm.weight                   | F16    | [4096]
model.layers.22.self_attn.k_proj.weight          -> blk.22.attn_k.weight                     | F16    | [1024, 4096]
model.layers.22.self_attn.o_proj.weight          -> blk.22.attn_output.weight                | F16    | [4096, 4096]
model.layers.22.self_attn.q_proj.weight          -> blk.22.attn_q.weight                     | F16    | [4096, 4096]
model.layers.22.self_attn.v_proj.weight          -> blk.22.attn_v.weight                     | F16    | [1024, 4096]
model.layers.23.input_layernorm.weight           -> blk.23.attn_norm.weight                  | F16    | [4096]
model.layers.23.mlp.down_proj.weight             -> blk.23.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.23.mlp.gate_proj.weight             -> blk.23.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.23.mlp.up_proj.weight               -> blk.23.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.23.post_attention_layernorm.weight  -> blk.23.ffn_norm.weight                   | F16    | [4096]
model.layers.23.self_attn.k_proj.weight          -> blk.23.attn_k.weight                     | F16    | [1024, 4096]
model.layers.23.self_attn.o_proj.weight          -> blk.23.attn_output.weight                | F16    | [4096, 4096]
model.layers.23.self_attn.q_proj.weight          -> blk.23.attn_q.weight                     | F16    | [4096, 4096]
model.layers.23.self_attn.v_proj.weight          -> blk.23.attn_v.weight                     | F16    | [1024, 4096]
model.layers.24.input_layernorm.weight           -> blk.24.attn_norm.weight                  | F16    | [4096]
model.layers.24.mlp.down_proj.weight             -> blk.24.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.24.mlp.gate_proj.weight             -> blk.24.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.24.mlp.up_proj.weight               -> blk.24.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.24.post_attention_layernorm.weight  -> blk.24.ffn_norm.weight                   | F16    | [4096]
model.layers.24.self_attn.k_proj.weight          -> blk.24.attn_k.weight                     | F16    | [1024, 4096]
model.layers.24.self_attn.o_proj.weight          -> blk.24.attn_output.weight                | F16    | [4096, 4096]
model.layers.24.self_attn.q_proj.weight          -> blk.24.attn_q.weight                     | F16    | [4096, 4096]
model.layers.24.self_attn.v_proj.weight          -> blk.24.attn_v.weight                     | F16    | [1024, 4096]
model.layers.25.input_layernorm.weight           -> blk.25.attn_norm.weight                  | F16    | [4096]
model.layers.25.mlp.down_proj.weight             -> blk.25.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.25.mlp.gate_proj.weight             -> blk.25.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.25.mlp.up_proj.weight               -> blk.25.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.25.post_attention_layernorm.weight  -> blk.25.ffn_norm.weight                   | F16    | [4096]
model.layers.25.self_attn.k_proj.weight          -> blk.25.attn_k.weight                     | F16    | [1024, 4096]
model.layers.25.self_attn.o_proj.weight          -> blk.25.attn_output.weight                | F16    | [4096, 4096]
model.layers.25.self_attn.q_proj.weight          -> blk.25.attn_q.weight                     | F16    | [4096, 4096]
model.layers.25.self_attn.v_proj.weight          -> blk.25.attn_v.weight                     | F16    | [1024, 4096]
model.layers.26.input_layernorm.weight           -> blk.26.attn_norm.weight                  | F16    | [4096]
model.layers.26.mlp.down_proj.weight             -> blk.26.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.26.mlp.gate_proj.weight             -> blk.26.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.26.mlp.up_proj.weight               -> blk.26.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.26.post_attention_layernorm.weight  -> blk.26.ffn_norm.weight                   | F16    | [4096]
model.layers.26.self_attn.k_proj.weight          -> blk.26.attn_k.weight                     | F16    | [1024, 4096]
model.layers.26.self_attn.o_proj.weight          -> blk.26.attn_output.weight                | F16    | [4096, 4096]
model.layers.26.self_attn.q_proj.weight          -> blk.26.attn_q.weight                     | F16    | [4096, 4096]
model.layers.26.self_attn.v_proj.weight          -> blk.26.attn_v.weight                     | F16    | [1024, 4096]
model.layers.27.input_layernorm.weight           -> blk.27.attn_norm.weight                  | F16    | [4096]
model.layers.27.mlp.down_proj.weight             -> blk.27.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.27.mlp.gate_proj.weight             -> blk.27.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.27.mlp.up_proj.weight               -> blk.27.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.27.post_attention_layernorm.weight  -> blk.27.ffn_norm.weight                   | F16    | [4096]
model.layers.27.self_attn.k_proj.weight          -> blk.27.attn_k.weight                     | F16    | [1024, 4096]
model.layers.27.self_attn.o_proj.weight          -> blk.27.attn_output.weight                | F16    | [4096, 4096]
model.layers.27.self_attn.q_proj.weight          -> blk.27.attn_q.weight                     | F16    | [4096, 4096]
model.layers.27.self_attn.v_proj.weight          -> blk.27.attn_v.weight                     | F16    | [1024, 4096]
model.layers.28.input_layernorm.weight           -> blk.28.attn_norm.weight                  | F16    | [4096]
model.layers.28.mlp.down_proj.weight             -> blk.28.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.28.mlp.gate_proj.weight             -> blk.28.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.28.mlp.up_proj.weight               -> blk.28.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.28.post_attention_layernorm.weight  -> blk.28.ffn_norm.weight                   | F16    | [4096]
model.layers.28.self_attn.k_proj.weight          -> blk.28.attn_k.weight                     | F16    | [1024, 4096]
model.layers.28.self_attn.o_proj.weight          -> blk.28.attn_output.weight                | F16    | [4096, 4096]
model.layers.28.self_attn.q_proj.weight          -> blk.28.attn_q.weight                     | F16    | [4096, 4096]
model.layers.28.self_attn.v_proj.weight          -> blk.28.attn_v.weight                     | F16    | [1024, 4096]
model.layers.29.input_layernorm.weight           -> blk.29.attn_norm.weight                  | F16    | [4096]
model.layers.29.mlp.down_proj.weight             -> blk.29.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.29.mlp.gate_proj.weight             -> blk.29.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.29.mlp.up_proj.weight               -> blk.29.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.29.post_attention_layernorm.weight  -> blk.29.ffn_norm.weight                   | F16    | [4096]
model.layers.29.self_attn.k_proj.weight          -> blk.29.attn_k.weight                     | F16    | [1024, 4096]
model.layers.29.self_attn.o_proj.weight          -> blk.29.attn_output.weight                | F16    | [4096, 4096]
model.layers.29.self_attn.q_proj.weight          -> blk.29.attn_q.weight                     | F16    | [4096, 4096]
model.layers.29.self_attn.v_proj.weight          -> blk.29.attn_v.weight                     | F16    | [1024, 4096]
model.layers.3.input_layernorm.weight            -> blk.3.attn_norm.weight                   | F16    | [4096]
model.layers.3.mlp.down_proj.weight              -> blk.3.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.3.mlp.gate_proj.weight              -> blk.3.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.3.mlp.up_proj.weight                -> blk.3.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.3.post_attention_layernorm.weight   -> blk.3.ffn_norm.weight                    | F16    | [4096]
model.layers.3.self_attn.k_proj.weight           -> blk.3.attn_k.weight                      | F16    | [1024, 4096]
model.layers.3.self_attn.o_proj.weight           -> blk.3.attn_output.weight                 | F16    | [4096, 4096]
model.layers.3.self_attn.q_proj.weight           -> blk.3.attn_q.weight                      | F16    | [4096, 4096]
model.layers.3.self_attn.v_proj.weight           -> blk.3.attn_v.weight                      | F16    | [1024, 4096]
model.layers.30.input_layernorm.weight           -> blk.30.attn_norm.weight                  | F16    | [4096]
model.layers.30.mlp.down_proj.weight             -> blk.30.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.30.mlp.gate_proj.weight             -> blk.30.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.30.mlp.up_proj.weight               -> blk.30.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.30.post_attention_layernorm.weight  -> blk.30.ffn_norm.weight                   | F16    | [4096]
model.layers.30.self_attn.k_proj.weight          -> blk.30.attn_k.weight                     | F16    | [1024, 4096]
model.layers.30.self_attn.o_proj.weight          -> blk.30.attn_output.weight                | F16    | [4096, 4096]
model.layers.30.self_attn.q_proj.weight          -> blk.30.attn_q.weight                     | F16    | [4096, 4096]
model.layers.30.self_attn.v_proj.weight          -> blk.30.attn_v.weight                     | F16    | [1024, 4096]
model.layers.31.input_layernorm.weight           -> blk.31.attn_norm.weight                  | F16    | [4096]
model.layers.31.mlp.down_proj.weight             -> blk.31.ffn_down.weight                   | F16    | [4096, 14336]
model.layers.31.mlp.gate_proj.weight             -> blk.31.ffn_gate.weight                   | F16    | [14336, 4096]
model.layers.31.mlp.up_proj.weight               -> blk.31.ffn_up.weight                     | F16    | [14336, 4096]
model.layers.31.post_attention_layernorm.weight  -> blk.31.ffn_norm.weight                   | F16    | [4096]
model.layers.31.self_attn.k_proj.weight          -> blk.31.attn_k.weight                     | F16    | [1024, 4096]
model.layers.31.self_attn.o_proj.weight          -> blk.31.attn_output.weight                | F16    | [4096, 4096]
model.layers.31.self_attn.q_proj.weight          -> blk.31.attn_q.weight                     | F16    | [4096, 4096]
model.layers.31.self_attn.v_proj.weight          -> blk.31.attn_v.weight                     | F16    | [1024, 4096]
model.layers.4.input_layernorm.weight            -> blk.4.attn_norm.weight                   | F16    | [4096]
model.layers.4.mlp.down_proj.weight              -> blk.4.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.4.mlp.gate_proj.weight              -> blk.4.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.4.mlp.up_proj.weight                -> blk.4.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.4.post_attention_layernorm.weight   -> blk.4.ffn_norm.weight                    | F16    | [4096]
model.layers.4.self_attn.k_proj.weight           -> blk.4.attn_k.weight                      | F16    | [1024, 4096]
model.layers.4.self_attn.o_proj.weight           -> blk.4.attn_output.weight                 | F16    | [4096, 4096]
model.layers.4.self_attn.q_proj.weight           -> blk.4.attn_q.weight                      | F16    | [4096, 4096]
model.layers.4.self_attn.v_proj.weight           -> blk.4.attn_v.weight                      | F16    | [1024, 4096]
model.layers.5.input_layernorm.weight            -> blk.5.attn_norm.weight                   | F16    | [4096]
model.layers.5.mlp.down_proj.weight              -> blk.5.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.5.mlp.gate_proj.weight              -> blk.5.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.5.mlp.up_proj.weight                -> blk.5.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.5.post_attention_layernorm.weight   -> blk.5.ffn_norm.weight                    | F16    | [4096]
model.layers.5.self_attn.k_proj.weight           -> blk.5.attn_k.weight                      | F16    | [1024, 4096]
model.layers.5.self_attn.o_proj.weight           -> blk.5.attn_output.weight                 | F16    | [4096, 4096]
model.layers.5.self_attn.q_proj.weight           -> blk.5.attn_q.weight                      | F16    | [4096, 4096]
model.layers.5.self_attn.v_proj.weight           -> blk.5.attn_v.weight                      | F16    | [1024, 4096]
model.layers.6.input_layernorm.weight            -> blk.6.attn_norm.weight                   | F16    | [4096]
model.layers.6.mlp.down_proj.weight              -> blk.6.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.6.mlp.gate_proj.weight              -> blk.6.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.6.mlp.up_proj.weight                -> blk.6.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.6.post_attention_layernorm.weight   -> blk.6.ffn_norm.weight                    | F16    | [4096]
model.layers.6.self_attn.k_proj.weight           -> blk.6.attn_k.weight                      | F16    | [1024, 4096]
model.layers.6.self_attn.o_proj.weight           -> blk.6.attn_output.weight                 | F16    | [4096, 4096]
model.layers.6.self_attn.q_proj.weight           -> blk.6.attn_q.weight                      | F16    | [4096, 4096]
model.layers.6.self_attn.v_proj.weight           -> blk.6.attn_v.weight                      | F16    | [1024, 4096]
model.layers.7.input_layernorm.weight            -> blk.7.attn_norm.weight                   | F16    | [4096]
model.layers.7.mlp.down_proj.weight              -> blk.7.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.7.mlp.gate_proj.weight              -> blk.7.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.7.mlp.up_proj.weight                -> blk.7.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.7.post_attention_layernorm.weight   -> blk.7.ffn_norm.weight                    | F16    | [4096]
model.layers.7.self_attn.k_proj.weight           -> blk.7.attn_k.weight                      | F16    | [1024, 4096]
model.layers.7.self_attn.o_proj.weight           -> blk.7.attn_output.weight                 | F16    | [4096, 4096]
model.layers.7.self_attn.q_proj.weight           -> blk.7.attn_q.weight                      | F16    | [4096, 4096]
model.layers.7.self_attn.v_proj.weight           -> blk.7.attn_v.weight                      | F16    | [1024, 4096]
model.layers.8.input_layernorm.weight            -> blk.8.attn_norm.weight                   | F16    | [4096]
model.layers.8.mlp.down_proj.weight              -> blk.8.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.8.mlp.gate_proj.weight              -> blk.8.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.8.mlp.up_proj.weight                -> blk.8.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.8.post_attention_layernorm.weight   -> blk.8.ffn_norm.weight                    | F16    | [4096]
model.layers.8.self_attn.k_proj.weight           -> blk.8.attn_k.weight                      | F16    | [1024, 4096]
model.layers.8.self_attn.o_proj.weight           -> blk.8.attn_output.weight                 | F16    | [4096, 4096]
model.layers.8.self_attn.q_proj.weight           -> blk.8.attn_q.weight                      | F16    | [4096, 4096]
model.layers.8.self_attn.v_proj.weight           -> blk.8.attn_v.weight                      | F16    | [1024, 4096]
model.layers.9.input_layernorm.weight            -> blk.9.attn_norm.weight                   | F16    | [4096]
model.layers.9.mlp.down_proj.weight              -> blk.9.ffn_down.weight                    | F16    | [4096, 14336]
model.layers.9.mlp.gate_proj.weight              -> blk.9.ffn_gate.weight                    | F16    | [14336, 4096]
model.layers.9.mlp.up_proj.weight                -> blk.9.ffn_up.weight                      | F16    | [14336, 4096]
model.layers.9.post_attention_layernorm.weight   -> blk.9.ffn_norm.weight                    | F16    | [4096]
model.layers.9.self_attn.k_proj.weight           -> blk.9.attn_k.weight                      | F16    | [1024, 4096]
model.layers.9.self_attn.o_proj.weight           -> blk.9.attn_output.weight                 | F16    | [4096, 4096]
model.layers.9.self_attn.q_proj.weight           -> blk.9.attn_q.weight                      | F16    | [4096, 4096]
model.layers.9.self_attn.v_proj.weight           -> blk.9.attn_v.weight                      | F16    | [1024, 4096]
model.norm.weight                                -> output_norm.weight                       | F16    | [4096]
Writing instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab.gguf, format 1
Padding vocab with 3 token(s) - <dummy00001> through <dummy00003>
gguf: This GGUF file is for Little Endian only
gguf: Setting special token type bos to 1
gguf: Setting special token type eos to 32000
gguf: Setting special token type unk to 0
gguf: Setting special token type pad to 32001
gguf: Setting add_bos_token to False
gguf: Setting add_eos_token to False
gguf: Setting chat_template to {% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>'+ '
' + message['content'] + '
'}}{% elif message['role'] == 'user' %}{{'<|user|>' + '
' + message['content'] + '
'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>' + '
' + message['content'] + '<|endoftext|>' + ('' if loop.last else '
')}}{% endif %}{% endfor %}
[  1/291] Writing tensor output.weight                          | size  32008 x   4096  | type F16  | T+   0
[  2/291] Writing tensor token_embd.weight                      | size  32008 x   4096  | type F16  | T+   0
[  3/291] Writing tensor blk.0.attn_norm.weight                 | size   4096           | type F32  | T+   0
[  4/291] Writing tensor blk.0.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+   0
[  5/291] Writing tensor blk.0.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+   0
[  6/291] Writing tensor blk.0.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+   0
[  7/291] Writing tensor blk.0.ffn_norm.weight                  | size   4096           | type F32  | T+   0
[  8/291] Writing tensor blk.0.attn_k.weight                    | size   1024 x   4096  | type F16  | T+   0
[  9/291] Writing tensor blk.0.attn_output.weight               | size   4096 x   4096  | type F16  | T+   1
[ 10/291] Writing tensor blk.0.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 11/291] Writing tensor blk.0.attn_v.weight                    | size   1024 x   4096  | type F16  | T+   1
[ 12/291] Writing tensor blk.1.attn_norm.weight                 | size   4096           | type F32  | T+   1
[ 13/291] Writing tensor blk.1.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+   1
[ 14/291] Writing tensor blk.1.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+   1
[ 15/291] Writing tensor blk.1.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+   1
[ 16/291] Writing tensor blk.1.ffn_norm.weight                  | size   4096           | type F32  | T+   1
[ 17/291] Writing tensor blk.1.attn_k.weight                    | size   1024 x   4096  | type F16  | T+   1
[ 18/291] Writing tensor blk.1.attn_output.weight               | size   4096 x   4096  | type F16  | T+   1
[ 19/291] Writing tensor blk.1.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 20/291] Writing tensor blk.1.attn_v.weight                    | size   1024 x   4096  | type F16  | T+   1
[ 21/291] Writing tensor blk.10.attn_norm.weight                | size   4096           | type F32  | T+   1
[ 22/291] Writing tensor blk.10.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   2
[ 23/291] Writing tensor blk.10.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   2
[ 24/291] Writing tensor blk.10.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   2
[ 25/291] Writing tensor blk.10.ffn_norm.weight                 | size   4096           | type F32  | T+   2
[ 26/291] Writing tensor blk.10.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   2
[ 27/291] Writing tensor blk.10.attn_output.weight              | size   4096 x   4096  | type F16  | T+   2
[ 28/291] Writing tensor blk.10.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   2
[ 29/291] Writing tensor blk.10.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   2
[ 30/291] Writing tensor blk.11.attn_norm.weight                | size   4096           | type F32  | T+   2
[ 31/291] Writing tensor blk.11.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   2
[ 32/291] Writing tensor blk.11.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   2
[ 33/291] Writing tensor blk.11.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   2
[ 34/291] Writing tensor blk.11.ffn_norm.weight                 | size   4096           | type F32  | T+   2
[ 35/291] Writing tensor blk.11.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   2
[ 36/291] Writing tensor blk.11.attn_output.weight              | size   4096 x   4096  | type F16  | T+   2
[ 37/291] Writing tensor blk.11.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   2
[ 38/291] Writing tensor blk.11.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   2
[ 39/291] Writing tensor blk.12.attn_norm.weight                | size   4096           | type F32  | T+   2
[ 40/291] Writing tensor blk.12.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   3
[ 41/291] Writing tensor blk.12.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   3
[ 42/291] Writing tensor blk.12.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   3
[ 43/291] Writing tensor blk.12.ffn_norm.weight                 | size   4096           | type F32  | T+   3
[ 44/291] Writing tensor blk.12.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   3
[ 45/291] Writing tensor blk.12.attn_output.weight              | size   4096 x   4096  | type F16  | T+   3
[ 46/291] Writing tensor blk.12.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   3
[ 47/291] Writing tensor blk.12.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   3
[ 48/291] Writing tensor blk.13.attn_norm.weight                | size   4096           | type F32  | T+   3
[ 49/291] Writing tensor blk.13.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   3
[ 50/291] Writing tensor blk.13.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   3
[ 51/291] Writing tensor blk.13.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   3
[ 52/291] Writing tensor blk.13.ffn_norm.weight                 | size   4096           | type F32  | T+   3
[ 53/291] Writing tensor blk.13.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   3
[ 54/291] Writing tensor blk.13.attn_output.weight              | size   4096 x   4096  | type F16  | T+   3
[ 55/291] Writing tensor blk.13.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   3
[ 56/291] Writing tensor blk.13.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   3
[ 57/291] Writing tensor blk.14.attn_norm.weight                | size   4096           | type F32  | T+   3
[ 58/291] Writing tensor blk.14.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   4
[ 59/291] Writing tensor blk.14.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   4
[ 60/291] Writing tensor blk.14.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   4
[ 61/291] Writing tensor blk.14.ffn_norm.weight                 | size   4096           | type F32  | T+   4
[ 62/291] Writing tensor blk.14.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   4
[ 63/291] Writing tensor blk.14.attn_output.weight              | size   4096 x   4096  | type F16  | T+   4
[ 64/291] Writing tensor blk.14.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   4
[ 65/291] Writing tensor blk.14.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   4
[ 66/291] Writing tensor blk.15.attn_norm.weight                | size   4096           | type F32  | T+   4
[ 67/291] Writing tensor blk.15.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   4
[ 68/291] Writing tensor blk.15.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   4
[ 69/291] Writing tensor blk.15.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   4
[ 70/291] Writing tensor blk.15.ffn_norm.weight                 | size   4096           | type F32  | T+   4
[ 71/291] Writing tensor blk.15.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   4
[ 72/291] Writing tensor blk.15.attn_output.weight              | size   4096 x   4096  | type F16  | T+   4
[ 73/291] Writing tensor blk.15.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   4
[ 74/291] Writing tensor blk.15.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   4
[ 75/291] Writing tensor blk.16.attn_norm.weight                | size   4096           | type F32  | T+   4
[ 76/291] Writing tensor blk.16.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   5
[ 77/291] Writing tensor blk.16.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   5
[ 78/291] Writing tensor blk.16.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   5
[ 79/291] Writing tensor blk.16.ffn_norm.weight                 | size   4096           | type F32  | T+   5
[ 80/291] Writing tensor blk.16.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   5
[ 81/291] Writing tensor blk.16.attn_output.weight              | size   4096 x   4096  | type F16  | T+   5
[ 82/291] Writing tensor blk.16.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   5
[ 83/291] Writing tensor blk.16.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   5
[ 84/291] Writing tensor blk.17.attn_norm.weight                | size   4096           | type F32  | T+   5
[ 85/291] Writing tensor blk.17.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   5
[ 86/291] Writing tensor blk.17.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   5
[ 87/291] Writing tensor blk.17.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   5
[ 88/291] Writing tensor blk.17.ffn_norm.weight                 | size   4096           | type F32  | T+   5
[ 89/291] Writing tensor blk.17.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   5
[ 90/291] Writing tensor blk.17.attn_output.weight              | size   4096 x   4096  | type F16  | T+   5
[ 91/291] Writing tensor blk.17.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   5
[ 92/291] Writing tensor blk.17.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   5
[ 93/291] Writing tensor blk.18.attn_norm.weight                | size   4096           | type F32  | T+   5
[ 94/291] Writing tensor blk.18.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   6
[ 95/291] Writing tensor blk.18.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   6
[ 96/291] Writing tensor blk.18.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   6
[ 97/291] Writing tensor blk.18.ffn_norm.weight                 | size   4096           | type F32  | T+   6
[ 98/291] Writing tensor blk.18.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   6
[ 99/291] Writing tensor blk.18.attn_output.weight              | size   4096 x   4096  | type F16  | T+   6
[100/291] Writing tensor blk.18.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   6
[101/291] Writing tensor blk.18.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   6
[102/291] Writing tensor blk.19.attn_norm.weight                | size   4096           | type F32  | T+   6
[103/291] Writing tensor blk.19.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   6
[104/291] Writing tensor blk.19.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   6
[105/291] Writing tensor blk.19.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   6
[106/291] Writing tensor blk.19.ffn_norm.weight                 | size   4096           | type F32  | T+   6
[107/291] Writing tensor blk.19.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   6
[108/291] Writing tensor blk.19.attn_output.weight              | size   4096 x   4096  | type F16  | T+   6
[109/291] Writing tensor blk.19.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   6
[110/291] Writing tensor blk.19.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   7
[111/291] Writing tensor blk.2.attn_norm.weight                 | size   4096           | type F32  | T+   7
[112/291] Writing tensor blk.2.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+   7
[113/291] Writing tensor blk.2.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+   7
[114/291] Writing tensor blk.2.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+   7
[115/291] Writing tensor blk.2.ffn_norm.weight                  | size   4096           | type F32  | T+   7
[116/291] Writing tensor blk.2.attn_k.weight                    | size   1024 x   4096  | type F16  | T+   7
[117/291] Writing tensor blk.2.attn_output.weight               | size   4096 x   4096  | type F16  | T+   7
[118/291] Writing tensor blk.2.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   7
[119/291] Writing tensor blk.2.attn_v.weight                    | size   1024 x   4096  | type F16  | T+   7
[120/291] Writing tensor blk.20.attn_norm.weight                | size   4096           | type F32  | T+   7
[121/291] Writing tensor blk.20.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   8
[122/291] Writing tensor blk.20.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   8
[123/291] Writing tensor blk.20.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   8
[124/291] Writing tensor blk.20.ffn_norm.weight                 | size   4096           | type F32  | T+   8
[125/291] Writing tensor blk.20.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   8
[126/291] Writing tensor blk.20.attn_output.weight              | size   4096 x   4096  | type F16  | T+   8
[127/291] Writing tensor blk.20.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   8
[128/291] Writing tensor blk.20.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   8
[129/291] Writing tensor blk.21.attn_norm.weight                | size   4096           | type F32  | T+   8
[130/291] Writing tensor blk.21.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+   9
[131/291] Writing tensor blk.21.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+   9
[132/291] Writing tensor blk.21.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+   9
[133/291] Writing tensor blk.21.ffn_norm.weight                 | size   4096           | type F32  | T+   9
[134/291] Writing tensor blk.21.attn_k.weight                   | size   1024 x   4096  | type F16  | T+   9
[135/291] Writing tensor blk.21.attn_output.weight              | size   4096 x   4096  | type F16  | T+   9
[136/291] Writing tensor blk.21.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   9
[137/291] Writing tensor blk.21.attn_v.weight                   | size   1024 x   4096  | type F16  | T+   9
[138/291] Writing tensor blk.22.attn_norm.weight                | size   4096           | type F32  | T+   9
[139/291] Writing tensor blk.22.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  10
[140/291] Writing tensor blk.22.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  10
[141/291] Writing tensor blk.22.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  10
[142/291] Writing tensor blk.22.ffn_norm.weight                 | size   4096           | type F32  | T+  10
[143/291] Writing tensor blk.22.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  10
[144/291] Writing tensor blk.22.attn_output.weight              | size   4096 x   4096  | type F16  | T+  10
[145/291] Writing tensor blk.22.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  10
[146/291] Writing tensor blk.22.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  10
[147/291] Writing tensor blk.23.attn_norm.weight                | size   4096           | type F32  | T+  10
[148/291] Writing tensor blk.23.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  11
[149/291] Writing tensor blk.23.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  11
[150/291] Writing tensor blk.23.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  11
[151/291] Writing tensor blk.23.ffn_norm.weight                 | size   4096           | type F32  | T+  11
[152/291] Writing tensor blk.23.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  11
[153/291] Writing tensor blk.23.attn_output.weight              | size   4096 x   4096  | type F16  | T+  11
[154/291] Writing tensor blk.23.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  11
[155/291] Writing tensor blk.23.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  11
[156/291] Writing tensor blk.24.attn_norm.weight                | size   4096           | type F32  | T+  11
[157/291] Writing tensor blk.24.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  12
[158/291] Writing tensor blk.24.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  12
[159/291] Writing tensor blk.24.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  12
[160/291] Writing tensor blk.24.ffn_norm.weight                 | size   4096           | type F32  | T+  12
[161/291] Writing tensor blk.24.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  12
[162/291] Writing tensor blk.24.attn_output.weight              | size   4096 x   4096  | type F16  | T+  12
[163/291] Writing tensor blk.24.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  12
[164/291] Writing tensor blk.24.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  12
[165/291] Writing tensor blk.25.attn_norm.weight                | size   4096           | type F32  | T+  12
[166/291] Writing tensor blk.25.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  13
[167/291] Writing tensor blk.25.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  13
[168/291] Writing tensor blk.25.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  13
[169/291] Writing tensor blk.25.ffn_norm.weight                 | size   4096           | type F32  | T+  13
[170/291] Writing tensor blk.25.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  13
[171/291] Writing tensor blk.25.attn_output.weight              | size   4096 x   4096  | type F16  | T+  13
[172/291] Writing tensor blk.25.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  13
[173/291] Writing tensor blk.25.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  13
[174/291] Writing tensor blk.26.attn_norm.weight                | size   4096           | type F32  | T+  13
[175/291] Writing tensor blk.26.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  14
[176/291] Writing tensor blk.26.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  14
[177/291] Writing tensor blk.26.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  14
[178/291] Writing tensor blk.26.ffn_norm.weight                 | size   4096           | type F32  | T+  14
[179/291] Writing tensor blk.26.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  14
[180/291] Writing tensor blk.26.attn_output.weight              | size   4096 x   4096  | type F16  | T+  14
[181/291] Writing tensor blk.26.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  14
[182/291] Writing tensor blk.26.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  14
[183/291] Writing tensor blk.27.attn_norm.weight                | size   4096           | type F32  | T+  14
[184/291] Writing tensor blk.27.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  15
[185/291] Writing tensor blk.27.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  15
[186/291] Writing tensor blk.27.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  15
[187/291] Writing tensor blk.27.ffn_norm.weight                 | size   4096           | type F32  | T+  15
[188/291] Writing tensor blk.27.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  15
[189/291] Writing tensor blk.27.attn_output.weight              | size   4096 x   4096  | type F16  | T+  15
[190/291] Writing tensor blk.27.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  15
[191/291] Writing tensor blk.27.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  15
[192/291] Writing tensor blk.28.attn_norm.weight                | size   4096           | type F32  | T+  15
[193/291] Writing tensor blk.28.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  16
[194/291] Writing tensor blk.28.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  16
[195/291] Writing tensor blk.28.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  16
[196/291] Writing tensor blk.28.ffn_norm.weight                 | size   4096           | type F32  | T+  16
[197/291] Writing tensor blk.28.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  16
[198/291] Writing tensor blk.28.attn_output.weight              | size   4096 x   4096  | type F16  | T+  16
[199/291] Writing tensor blk.28.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  16
[200/291] Writing tensor blk.28.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  16
[201/291] Writing tensor blk.29.attn_norm.weight                | size   4096           | type F32  | T+  16
[202/291] Writing tensor blk.29.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  17
[203/291] Writing tensor blk.29.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  17
[204/291] Writing tensor blk.29.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  17
[205/291] Writing tensor blk.29.ffn_norm.weight                 | size   4096           | type F32  | T+  17
[206/291] Writing tensor blk.29.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  17
[207/291] Writing tensor blk.29.attn_output.weight              | size   4096 x   4096  | type F16  | T+  17
[208/291] Writing tensor blk.29.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  17
[209/291] Writing tensor blk.29.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  17
[210/291] Writing tensor blk.3.attn_norm.weight                 | size   4096           | type F32  | T+  17
[211/291] Writing tensor blk.3.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  18
[212/291] Writing tensor blk.3.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  18
[213/291] Writing tensor blk.3.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  18
[214/291] Writing tensor blk.3.ffn_norm.weight                  | size   4096           | type F32  | T+  18
[215/291] Writing tensor blk.3.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  18
[216/291] Writing tensor blk.3.attn_output.weight               | size   4096 x   4096  | type F16  | T+  18
[217/291] Writing tensor blk.3.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  18
[218/291] Writing tensor blk.3.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  18
[219/291] Writing tensor blk.30.attn_norm.weight                | size   4096           | type F32  | T+  18
[220/291] Writing tensor blk.30.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  19
[221/291] Writing tensor blk.30.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  19
[222/291] Writing tensor blk.30.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  19
[223/291] Writing tensor blk.30.ffn_norm.weight                 | size   4096           | type F32  | T+  19
[224/291] Writing tensor blk.30.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  19
[225/291] Writing tensor blk.30.attn_output.weight              | size   4096 x   4096  | type F16  | T+  19
[226/291] Writing tensor blk.30.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  19
[227/291] Writing tensor blk.30.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  19
[228/291] Writing tensor blk.31.attn_norm.weight                | size   4096           | type F32  | T+  19
[229/291] Writing tensor blk.31.ffn_down.weight                 | size   4096 x  14336  | type F16  | T+  20
[230/291] Writing tensor blk.31.ffn_gate.weight                 | size  14336 x   4096  | type F16  | T+  20
[231/291] Writing tensor blk.31.ffn_up.weight                   | size  14336 x   4096  | type F16  | T+  20
[232/291] Writing tensor blk.31.ffn_norm.weight                 | size   4096           | type F32  | T+  20
[233/291] Writing tensor blk.31.attn_k.weight                   | size   1024 x   4096  | type F16  | T+  20
[234/291] Writing tensor blk.31.attn_output.weight              | size   4096 x   4096  | type F16  | T+  20
[235/291] Writing tensor blk.31.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  20
[236/291] Writing tensor blk.31.attn_v.weight                   | size   1024 x   4096  | type F16  | T+  20
[237/291] Writing tensor blk.4.attn_norm.weight                 | size   4096           | type F32  | T+  20
[238/291] Writing tensor blk.4.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  21
[239/291] Writing tensor blk.4.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  21
[240/291] Writing tensor blk.4.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  21
[241/291] Writing tensor blk.4.ffn_norm.weight                  | size   4096           | type F32  | T+  21
[242/291] Writing tensor blk.4.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  21
[243/291] Writing tensor blk.4.attn_output.weight               | size   4096 x   4096  | type F16  | T+  21
[244/291] Writing tensor blk.4.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  21
[245/291] Writing tensor blk.4.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  21
[246/291] Writing tensor blk.5.attn_norm.weight                 | size   4096           | type F32  | T+  21
[247/291] Writing tensor blk.5.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  22
[248/291] Writing tensor blk.5.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  22
[249/291] Writing tensor blk.5.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  22
[250/291] Writing tensor blk.5.ffn_norm.weight                  | size   4096           | type F32  | T+  22
[251/291] Writing tensor blk.5.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  22
[252/291] Writing tensor blk.5.attn_output.weight               | size   4096 x   4096  | type F16  | T+  22
[253/291] Writing tensor blk.5.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  22
[254/291] Writing tensor blk.5.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  22
[255/291] Writing tensor blk.6.attn_norm.weight                 | size   4096           | type F32  | T+  22
[256/291] Writing tensor blk.6.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  23
[257/291] Writing tensor blk.6.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  23
[258/291] Writing tensor blk.6.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  23
[259/291] Writing tensor blk.6.ffn_norm.weight                  | size   4096           | type F32  | T+  23
[260/291] Writing tensor blk.6.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  23
[261/291] Writing tensor blk.6.attn_output.weight               | size   4096 x   4096  | type F16  | T+  23
[262/291] Writing tensor blk.6.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  23
[263/291] Writing tensor blk.6.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  23
[264/291] Writing tensor blk.7.attn_norm.weight                 | size   4096           | type F32  | T+  23
[265/291] Writing tensor blk.7.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  24
[266/291] Writing tensor blk.7.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  24
[267/291] Writing tensor blk.7.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  24
[268/291] Writing tensor blk.7.ffn_norm.weight                  | size   4096           | type F32  | T+  24
[269/291] Writing tensor blk.7.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  24
[270/291] Writing tensor blk.7.attn_output.weight               | size   4096 x   4096  | type F16  | T+  24
[271/291] Writing tensor blk.7.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  24
[272/291] Writing tensor blk.7.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  24
[273/291] Writing tensor blk.8.attn_norm.weight                 | size   4096           | type F32  | T+  24
[274/291] Writing tensor blk.8.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  25
[275/291] Writing tensor blk.8.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  25
[276/291] Writing tensor blk.8.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  25
[277/291] Writing tensor blk.8.ffn_norm.weight                  | size   4096           | type F32  | T+  25
[278/291] Writing tensor blk.8.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  25
[279/291] Writing tensor blk.8.attn_output.weight               | size   4096 x   4096  | type F16  | T+  25
[280/291] Writing tensor blk.8.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  25
[281/291] Writing tensor blk.8.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  25
[282/291] Writing tensor blk.9.attn_norm.weight                 | size   4096           | type F32  | T+  25
[283/291] Writing tensor blk.9.ffn_down.weight                  | size   4096 x  14336  | type F16  | T+  26
[284/291] Writing tensor blk.9.ffn_gate.weight                  | size  14336 x   4096  | type F16  | T+  26
[285/291] Writing tensor blk.9.ffn_up.weight                    | size  14336 x   4096  | type F16  | T+  26
[286/291] Writing tensor blk.9.ffn_norm.weight                  | size   4096           | type F32  | T+  26
[287/291] Writing tensor blk.9.attn_k.weight                    | size   1024 x   4096  | type F16  | T+  26
[288/291] Writing tensor blk.9.attn_output.weight               | size   4096 x   4096  | type F16  | T+  26
[289/291] Writing tensor blk.9.attn_q.weight                    | size   4096 x   4096  | type F16  | T+  26
[290/291] Writing tensor blk.9.attn_v.weight                    | size   1024 x   4096  | type F16  | T+  26
[291/291] Writing tensor output_norm.weight                     | size   4096           | type F32  | T+  26
Wrote instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab.gguf
INFO 2024-06-20 18:50:54,938 lab.py:1362 deleting safetensors files from instructlab-merlinite-7b-lab-trained...
main: build = 1 (784e11d)
main: built with Apple clang version 15.0.0 (clang-1500.0.40.1) for arm64-apple-darwin23.4.0
main: quantizing 'instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab.gguf' to 'instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf' as Q4_K_M
llama_model_loader: loaded meta data with 23 key-value pairs and 291 tensors from instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = .
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                       llama.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11:                          general.file_type u32              = 1
llama_model_loader: - kv  12:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32008]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32008]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32008]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 32000
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32              = 32001
llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {% for message in messages %}{% if me...
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type  f16:  226 tensors
llama_model_quantize_internal: meta size = 735584 bytes
[   1/ 291]                        output.weight - [ 4096, 32008,     1,     1], type =    f16, converting to q6_K .. size =   250.06 MiB ->   102.56 MiB
[   2/ 291]                    token_embd.weight - [ 4096, 32008,     1,     1], type =    f16, converting to q4_K .. size =   250.06 MiB ->    70.33 MiB
[   3/ 291]               blk.0.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[   4/ 291]                blk.0.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[   5/ 291]                blk.0.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[   6/ 291]                  blk.0.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[   7/ 291]                blk.0.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[   8/ 291]                  blk.0.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[   9/ 291]             blk.0.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  10/ 291]                  blk.0.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  11/ 291]                  blk.0.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[  12/ 291]               blk.1.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  13/ 291]                blk.1.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[  14/ 291]                blk.1.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  15/ 291]                  blk.1.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  16/ 291]                blk.1.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  17/ 291]                  blk.1.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  18/ 291]             blk.1.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  19/ 291]                  blk.1.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  20/ 291]                  blk.1.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[  21/ 291]              blk.10.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  22/ 291]               blk.10.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[  23/ 291]               blk.10.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  24/ 291]                 blk.10.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  25/ 291]               blk.10.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  26/ 291]                 blk.10.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  27/ 291]            blk.10.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  28/ 291]                 blk.10.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  29/ 291]                 blk.10.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[  30/ 291]              blk.11.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  31/ 291]               blk.11.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[  32/ 291]               blk.11.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  33/ 291]                 blk.11.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  34/ 291]               blk.11.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  35/ 291]                 blk.11.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  36/ 291]            blk.11.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  37/ 291]                 blk.11.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  38/ 291]                 blk.11.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[  39/ 291]              blk.12.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  40/ 291]               blk.12.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  41/ 291]               blk.12.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  42/ 291]                 blk.12.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  43/ 291]               blk.12.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  44/ 291]                 blk.12.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  45/ 291]            blk.12.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  46/ 291]                 blk.12.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  47/ 291]                 blk.12.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  48/ 291]              blk.13.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  49/ 291]               blk.13.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  50/ 291]               blk.13.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  51/ 291]                 blk.13.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  52/ 291]               blk.13.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  53/ 291]                 blk.13.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  54/ 291]            blk.13.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  55/ 291]                 blk.13.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  56/ 291]                 blk.13.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  57/ 291]              blk.14.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  58/ 291]               blk.14.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[  59/ 291]               blk.14.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  60/ 291]                 blk.14.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  61/ 291]               blk.14.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  62/ 291]                 blk.14.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  63/ 291]            blk.14.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  64/ 291]                 blk.14.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  65/ 291]                 blk.14.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[  66/ 291]              blk.15.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  67/ 291]               blk.15.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  68/ 291]               blk.15.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  69/ 291]                 blk.15.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  70/ 291]               blk.15.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  71/ 291]                 blk.15.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  72/ 291]            blk.15.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  73/ 291]                 blk.15.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  74/ 291]                 blk.15.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  75/ 291]              blk.16.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  76/ 291]               blk.16.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  77/ 291]               blk.16.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  78/ 291]                 blk.16.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  79/ 291]               blk.16.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  80/ 291]                 blk.16.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  81/ 291]            blk.16.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  82/ 291]                 blk.16.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  83/ 291]                 blk.16.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  84/ 291]              blk.17.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  85/ 291]               blk.17.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[  86/ 291]               blk.17.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  87/ 291]                 blk.17.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  88/ 291]               blk.17.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  89/ 291]                 blk.17.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  90/ 291]            blk.17.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  91/ 291]                 blk.17.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[  92/ 291]                 blk.17.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[  93/ 291]              blk.18.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  94/ 291]               blk.18.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  95/ 291]               blk.18.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  96/ 291]                 blk.18.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[  97/ 291]               blk.18.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[  98/ 291]                 blk.18.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[  99/ 291]            blk.18.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 100/ 291]                 blk.18.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 101/ 291]                 blk.18.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 102/ 291]              blk.19.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 103/ 291]               blk.19.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 104/ 291]               blk.19.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 105/ 291]                 blk.19.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 106/ 291]               blk.19.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 107/ 291]                 blk.19.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 108/ 291]            blk.19.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 109/ 291]                 blk.19.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 110/ 291]                 blk.19.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 111/ 291]               blk.2.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 112/ 291]                blk.2.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 113/ 291]                blk.2.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 114/ 291]                  blk.2.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 115/ 291]                blk.2.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 116/ 291]                  blk.2.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 117/ 291]             blk.2.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 118/ 291]                  blk.2.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 119/ 291]                  blk.2.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 120/ 291]              blk.20.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 121/ 291]               blk.20.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 122/ 291]               blk.20.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 123/ 291]                 blk.20.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 124/ 291]               blk.20.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 125/ 291]                 blk.20.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 126/ 291]            blk.20.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 127/ 291]                 blk.20.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 128/ 291]                 blk.20.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 129/ 291]              blk.21.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 130/ 291]               blk.21.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 131/ 291]               blk.21.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 132/ 291]                 blk.21.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 133/ 291]               blk.21.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 134/ 291]                 blk.21.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 135/ 291]            blk.21.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 136/ 291]                 blk.21.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 137/ 291]                 blk.21.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 138/ 291]              blk.22.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 139/ 291]               blk.22.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 140/ 291]               blk.22.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 141/ 291]                 blk.22.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 142/ 291]               blk.22.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 143/ 291]                 blk.22.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 144/ 291]            blk.22.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 145/ 291]                 blk.22.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 146/ 291]                 blk.22.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 147/ 291]              blk.23.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 148/ 291]               blk.23.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 149/ 291]               blk.23.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 150/ 291]                 blk.23.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 151/ 291]               blk.23.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 152/ 291]                 blk.23.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 153/ 291]            blk.23.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 154/ 291]                 blk.23.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 155/ 291]                 blk.23.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 156/ 291]              blk.24.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 157/ 291]               blk.24.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 158/ 291]               blk.24.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 159/ 291]                 blk.24.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 160/ 291]               blk.24.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 161/ 291]                 blk.24.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 162/ 291]            blk.24.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 163/ 291]                 blk.24.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 164/ 291]                 blk.24.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 165/ 291]              blk.25.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 166/ 291]               blk.25.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 167/ 291]               blk.25.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 168/ 291]                 blk.25.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 169/ 291]               blk.25.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 170/ 291]                 blk.25.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 171/ 291]            blk.25.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 172/ 291]                 blk.25.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 173/ 291]                 blk.25.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 174/ 291]              blk.26.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 175/ 291]               blk.26.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 176/ 291]               blk.26.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 177/ 291]                 blk.26.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 178/ 291]               blk.26.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 179/ 291]                 blk.26.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 180/ 291]            blk.26.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 181/ 291]                 blk.26.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 182/ 291]                 blk.26.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 183/ 291]              blk.27.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 184/ 291]               blk.27.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 185/ 291]               blk.27.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 186/ 291]                 blk.27.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 187/ 291]               blk.27.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 188/ 291]                 blk.27.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 189/ 291]            blk.27.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 190/ 291]                 blk.27.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 191/ 291]                 blk.27.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 192/ 291]              blk.28.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 193/ 291]               blk.28.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 194/ 291]               blk.28.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 195/ 291]                 blk.28.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 196/ 291]               blk.28.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 197/ 291]                 blk.28.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 198/ 291]            blk.28.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 199/ 291]                 blk.28.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 200/ 291]                 blk.28.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 201/ 291]              blk.29.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 202/ 291]               blk.29.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 203/ 291]               blk.29.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 204/ 291]                 blk.29.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 205/ 291]               blk.29.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 206/ 291]                 blk.29.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 207/ 291]            blk.29.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 208/ 291]                 blk.29.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 209/ 291]                 blk.29.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 210/ 291]               blk.3.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 211/ 291]                blk.3.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 212/ 291]                blk.3.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 213/ 291]                  blk.3.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 214/ 291]                blk.3.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 215/ 291]                  blk.3.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 216/ 291]             blk.3.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 217/ 291]                  blk.3.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 218/ 291]                  blk.3.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 219/ 291]              blk.30.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 220/ 291]               blk.30.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 221/ 291]               blk.30.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 222/ 291]                 blk.30.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 223/ 291]               blk.30.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 224/ 291]                 blk.30.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 225/ 291]            blk.30.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 226/ 291]                 blk.30.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 227/ 291]                 blk.30.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 228/ 291]              blk.31.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 229/ 291]               blk.31.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 230/ 291]               blk.31.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 231/ 291]                 blk.31.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 232/ 291]               blk.31.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 233/ 291]                 blk.31.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 234/ 291]            blk.31.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 235/ 291]                 blk.31.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 236/ 291]                 blk.31.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 237/ 291]               blk.4.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 238/ 291]                blk.4.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 239/ 291]                blk.4.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 240/ 291]                  blk.4.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 241/ 291]                blk.4.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 242/ 291]                  blk.4.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 243/ 291]             blk.4.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 244/ 291]                  blk.4.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 245/ 291]                  blk.4.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 246/ 291]               blk.5.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 247/ 291]                blk.5.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 248/ 291]                blk.5.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 249/ 291]                  blk.5.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 250/ 291]                blk.5.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 251/ 291]                  blk.5.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 252/ 291]             blk.5.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 253/ 291]                  blk.5.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 254/ 291]                  blk.5.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 255/ 291]               blk.6.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 256/ 291]                blk.6.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 257/ 291]                blk.6.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 258/ 291]                  blk.6.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 259/ 291]                blk.6.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 260/ 291]                  blk.6.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 261/ 291]             blk.6.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 262/ 291]                  blk.6.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 263/ 291]                  blk.6.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 264/ 291]               blk.7.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 265/ 291]                blk.7.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 266/ 291]                blk.7.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 267/ 291]                  blk.7.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 268/ 291]                blk.7.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 269/ 291]                  blk.7.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 270/ 291]             blk.7.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 271/ 291]                  blk.7.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 272/ 291]                  blk.7.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 273/ 291]               blk.8.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 274/ 291]                blk.8.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 275/ 291]                blk.8.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 276/ 291]                  blk.8.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 277/ 291]                blk.8.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 278/ 291]                  blk.8.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 279/ 291]             blk.8.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 280/ 291]                  blk.8.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 281/ 291]                  blk.8.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 282/ 291]               blk.9.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 283/ 291]                blk.9.ffn_down.weight - [14336,  4096,     1,     1], type =    f16, converting to q6_K .. size =   112.00 MiB ->    45.94 MiB
[ 284/ 291]                blk.9.ffn_gate.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 285/ 291]                  blk.9.ffn_up.weight - [ 4096, 14336,     1,     1], type =    f16, converting to q4_K .. size =   112.00 MiB ->    31.50 MiB
[ 286/ 291]                blk.9.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
[ 287/ 291]                  blk.9.attn_k.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q4_K .. size =     8.00 MiB ->     2.25 MiB
[ 288/ 291]             blk.9.attn_output.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 289/ 291]                  blk.9.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_K .. size =    32.00 MiB ->     9.00 MiB
[ 290/ 291]                  blk.9.attn_v.weight - [ 4096,  1024,     1,     1], type =    f16, converting to q6_K .. size =     8.00 MiB ->     3.28 MiB
[ 291/ 291]                   output_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
llama_model_quantize_internal: model size  = 13813.14 MB
llama_model_quantize_internal: quant size  =  4165.41 MB
INFO 2024-06-20 18:51:53,586 lab.py:1372 deleting instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab.gguf...
```

运行该命令后，所有权重和适配器都将转换为量化的 gguf 模型。目录中将创建一个目录`ilab`，其名称类似于：`instructlab-merlinite-7b-lab-trained`。

<aside>
💡 `Llama.cpp` 的主要优势在于其静态分析和代码生成功能。将深度学习模型量化，并且在低配置机型（没有GPU）上就可以做到模型推理。合并权重，生成全量模型权重。

</aside>

### **验证测试训练好的模型**

运行新训练的模型

`ilab serve --model-path instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf --num-threads 14 --max-ctx-size 20480`

在生成合成数据集、训练模型、测试模型的第二个终端中，运行以下命令与模型聊天：

`ilab chat -gm -m instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf`

<aside>
💡 Greedy decoding 模型每一步都选择具有最高概率的词作为输出，而不考虑整个句子的全局最优性。在每一步生成时，模型根据当前输入和上下文计算每个可能词的概率，然后选择概率最高的词作为当前步的输出。计算速度很快，不需要复杂的搜索算法，因此在实际应用中能快速生成文本，但是通常缺乏多样性，可能会产生重复或单调的句子。只考虑局部最优，可能会错过一些更好的句子结构或上下文，导致生成的文本在全局上不是最优的。总的来说，greedy decoding 是一种简单快速但可能不完美的文本生成策略，适用于对生成速度有较高要求但对文本多样性要求不高的应用场景。

</aside>

---

**在没有生成新的训练数据集之前，用下面的问题测试，返回出的答案，可以看出来信息不正确，不能达到预期**
`ilab chat -m models/granite-7b-lab-Q4_K_M.gguf`

**Who starred in the movie Oppenheimer?**

The movie "Oppenheimer" is **an upcoming film** adaptation of the Pulitzer Prize-winning biography "American Prometheus: The Life of J. Robert Oppenheimer" by Kira Davis. Cillian Murphy has been cast as the title character, J. Robert Oppenheimer. Other notable stars who have joined the cast include Matt Damon as Leslie Groves, the head of the Yale project and the father of Oppenheimer's wife, Emily Blunt as Katherine Manning Oppenheimer, the wife of J. Robert Oppenheimer, and Alden Ehrenreich as Donald Oppenheimer, their son.

The film is directed by Christopher Nolan, known for his work on blockbuster movies like "Batman Begins," "The Dark Knight," and "Interstellar." "Oppenheimer" is set to explore the life of J. Robert Oppenheimer, the man who led the project that developed the first atomic bomb during World War II. The movie is expected to delve into the moral complexities of nuclear power and the human cost of scientific progress.

While there isn't much information available about the plot yet, I can provide you with some context on the historical figures and events that "Oppenheimer" might depict. The film will likely cover Oppenheimer's early life, his education at the prestigious Institute for Advanced Studies in Princeton, New Jersey, and his work at the Los Alamos National Laboratory during the Manhattan Project. Additionally, the movie may explore Oppenheimer's struggles with the moral implications of nuclear weapons and his eventual condemnation by the U.S. government during the Red Scare.

**Stay tuned for more updates** on "Oppenheimer" as filming progresses and **more information becomes available**. If you have any questions about the cast, directors, or historical context of the movie, please let me know!

**What are the release dates for Oppenheimer movie?**

As of my last update, the release date for "Oppenheimer" **is not yet set**. The film was announced in 2021, and production has recently started. Christopher Nolan, the director, often takes his time to perfect his films before releasing them. Given the complexity of the subject matter and the intricate plot, it's likely that "Oppenheimer" will require a considerable amount of post-production work.

**Once filming is complete and the movie is ready for release**, the production team will announce an official date. Typically, this happens several months before the release date. I recommend checking official film news sources and the film's official website for the most up-to-date information on the release date.

In the meantime, you can look forward to other movies released by Christopher Nolan or explore other fascinating historical figures and events that have been brought to the big screen. For example, "Dunkirk" (2017) and "Interstellar" (2014) are two of Nolan's previous films that showcase his ability to create immersive and thought-provoking cinematic experiences.

**在加入了knowledge和上下文参考数据，并且使用训练的新模型之后，答案基本可以达到预期**
 `ilab chat -gm -m instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf`

**Who starred in the movie Oppenheimer?**

Sean Penn portrayed J. Robert Oppenheimer in the movie "Oppenheimer." He was nominated for an Academy Award │
│ for Best Actor for his performance.

**What are the release dates for Oppenheimer movie?**

The film Oppenheimer was released on March 17, 2023, in the United States after being shown at the Cannes Film Festival in 2022. It has been available to stream on HBO Max since its release.

```bash

╭─────────────────────────────────────────────────────────────────────────────────────────────── system ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ Welcome to InstructLab Chat w/ INSTRUCTLAB-MERLINITE-7B-LAB-TRAINED/INSTRUCTLAB-MERLINITE-7B-LAB-Q4_K_M.GGUF (type /h for help)                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>> Who starred in the movie Oppenheimer?                                                                                                                                                   [S][default]
╭─────────────────────────────────────────────────────────── instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf ────────────────────────────────────────────────────────────╮
│ Oppenheimer was written and directed by James Schamus, and it features Daniel Day-Lewis in the title role. Other notable actors in this film include Emily Blunt as Katherine "Kitty" Oppenheimer,   │
│ Robert Wagner as General Leslie Groves, and Michael Nyqvist as Albert Einstein. This movie is a historical drama that depicts the life of J. Robert Oppenheimer during his time at Los Alamos, where │
│ he led the development of the atomic bomb during World War II.                                                                                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 2.466 seconds ─╯
>>> What are the release dates for Oppenheimer movie?                                                                                                                                       [S][default]
╭─────────────────────────────────────────────────────────── instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf ────────────────────────────────────────────────────────────╮
│ The Oppenheimer movie was released in 2023. It premiered on March 14th and started to stream on Hulu on April 7th.                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.829 seconds ─╯
>>>
```

参考文章：

[IBM Developer](https://developer.ibm.com/tutorials/awb-train-open-source-llms-collected-knowledge-instructlab/)
