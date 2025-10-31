# DaSHLab SysML Induction Assignment - Level 1

Hello and welcome to **Level 1** of the DaSHLab SysML (Systems for ML) Induction Assignment!

In this assignment, we'll be reading about the basic building blocks of LLMs (like ChatGPT or Gemini) - a Transformer. This won't be a very difficult assignment to follow through, and you're allowed to use ChatGPT, Gemini and the likes to get a better understanding of the concepts you'll learn.

## But what is a Transformer?
Imagine you’re reading a book. You don’t just understand each word by itself -  you look at the whole sentence and think about how the words connect. That’s exactly what a Transformer does! It’s a type of machine learning model that helps computers understand language (and even images or sounds) by looking at all the information at once, not just one word at a time.

Transformers were introduced in 2017 in a paper called “Attention Is All You Need.” The key idea behind them is something called attention - it’s like giving more focus to the important words in a sentence, just like you’d pay more attention to key clues while focusing on your lectures right before exams (in case the IC spoils a Compre question, you never know!). This simple but powerful idea revolutionalized how AI models learn and made it possible for tools like ChatGPT, BERT, and many others to understand and generate human-like text.

## Why is this "Transformer" important?

Transformers are the brains behind tools like ChatGPT! Large Language Models (LLMs) take the Transformer’s idea of “attention” and scale it up massively (upto hundreds of billions of parameters!). Instead of just reading a short sentence, they read huge amounts of text from books, websites, and articles, learning how words, phrases, and ideas connect. The model’s attention mechanism helps it focus on the right parts of text when generating answers - just like you remembering key details from earlier chapters while summarizing a story.

By stacking many Transformer blocks together, LLMs can understand context, reason about meaning, and even generate creative responses. Each layer refines the model’s understanding a little more, until it can predict the next word in a sentence with impressive accuracy. In short, Transformers are the engine, and the massive training data and fine-tuning make LLMs the supercharged car that can chat, write, and think almost like a human!

## What is this assignment about?

For this assignment, we'll be looking at the Transformer architecture proposed by the 2017 paper "Attention is all you need". We'll try to understand how the attention mechanism works, and how LLMs are able to use attention blocks and neural network layers to process your prompts, and thus have coherent (and often deep, intellectual) conversations with you! 

This assignment will mainly focus on processing textual data instead of multimodal data (like images, videos, etc.). This is to help simplify the learning curve, and make it easier and intuitive to understand what's happening beneath the hood.

## Resources
Here's a list of resources that are most relevant for you to dive into this topic:

- 3Blue1Brown's YouTube playlist on **Neural Networks and Transformer Architecture**: [Link](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- A. Vaswani et al., **“Attention Is All You Need”**, 2017 - [Link](https://arxiv.org/abs/1706.03762)
- Jay Alammar's blog post on **The Illustrated Transformer** - [Link](https://jalammar.github.io/illustrated-transformer/)
- Jay Alammar's blog post on **The Illustrated GPT-2** - [Link](https://jalammar.github.io/illustrated-gpt2/)
- **What is Quantization in LLM** by Nithin Devanand - [Link](https://medium.com/@techresearchspace/what-is-quantization-in-llm-01ba61968a51)

Feel free to refer to any other resources that you think are more comfortable for you to learn from. For the purpose of this assignment, the resources given above should suffice.

## Quick Note
This assignment helps you develop the basics of how Transformers work in general. At DaSHLab, we use these principles to introduce **Systems-Level Optimizations** to Transformers-based models, and compare the figures we obtain with the original, non-optimized version of the model, and with other state-of-the-art optimizations.

## The Quiz (!?)
Congratulations on finishing the first step of this assignment! In order to check your understanding of the concepts you've recently learned, we've designed the below quiz for you to answer. This quiz, besides checking your understanding, also encourages you to look at Transformers from a Systems perspective, rather than a purely ML perspective. 

You're supposed to answer these questions in not more than 60-80 words. Please do not be intimidated by the length of the question, as almost all questions are designed to be deliberately long so that they can direct your thought process better.

Fill the answers to the given questions in this [Google Form](https://forms.gle/tJ9Krj1NXDw85wBj7)

For any queries, contact me (Atharva Pandit) or any one of us from the SysML induction team. 

Contact: +91 8591528972

---

### Q1  
This question is about the Feed-forward neural network (FFNN) block present in the transformer architecture.  
You've noticed that the FFNN has two types of layers, namely **Linear** and **ReLU** (besides LayerNorm and other layers, which we omit here).  
The neural network applies weights and biases in the linear layer, and ReLU layer introduces a non-linearity.  

Why do you think this non-linearity is needed?  
Are we better off not using ReLU (and thus saving some compute)?  
If not, are there any alternatives to ReLU that could be thought of? (they need not be more efficient, just alternatives are enough)

---

### Q2  
During your reading for the assignment, you come across **quantization**, a way to reduce the LLM model's size by expressing its weights as 16-bit Floating-Point numbers or 8-bit integers.  
You decide to apply quantization to the LLM you're working with, so that you can fit a larger LLM into a smaller GPU VRAM and get better responses.  

What could be the possible issues you face with this approach?

---

### Q3  
You've read about the attention mechanism for this assignment.  
Current LLM models implement a lot of attention heads (**multi-headed attention** or **MHA**) instead of having one big attention head.  

Why do you think this is the case?

---

### Q4 – Q5  
Based on the following case study:

Pranav and Soham are new to machine learning, and have recently learned about transformers and neural networks.  
They want to start off by implementing a neural network that can help identify handwritten digits, with training data obtained from the **MNIST dataset**.  
They decide to have a neural network architecture that looks like this:

> 28 × 28 image → 784-dimensional vector → passes through a layer of 784 neurons and compresses it to 256 dimensions → apply ReLU →  
> Pass through another layer, compressing it to 128 dimensions → ReLU →  
> Another layer that compresses it to 10 dimensions → apply softmax → Make a prediction

---

### Q4  
They've now decided to begin training on the dataset, and split it into train, validation and test sets.  
For the loss function, they'll be using softmax regression, and optimizer would be the **SGD optimizer**.  

Pranav decides that the learning rate should be set to **1** so that the training ends quicker, and begins training the NN.  
However, the training seems to continue for a long time, with no significant improvements in both training and validation accuracy.  
Moreover, the values for accuracy seem to be oscillatory (e.g., the accuracy would jump from 56% to 59%, and back to 56% again after successive iterations).  

Why do you think this could be the case?

---

### Q5  
Thanks to your help, they were able to figure out the problem!  
However, their NN seems to perform very poorly (~60% accuracy) on the prediction task.  
Soham suggests that adding more neurons to the intermediate layers would help the model capture relationships better.  
Using this suggestion, Pranav and Soham notice that the model's performance on the training set has improved, but the model is performing poorly on the validation set.  

Why has this happened, and how could this be mitigated?

---

### Q6 – Q7  
Based on the following case study:

Vimarsh is hosting a **LLM** called *DaSHGPT* on his small cluster of GPUs for about 70 people.  
The LLM uses the classic transformer architecture that you've learnt about (embedding, attention heads, FFNN, softmax).  
Vimarsh also knows that GPUs are good at processing things in parallel, and decides to leverage this by parallelizing the pipeline as much as he can.

---

### Q6  
What are some parallelization techniques he could use to speed up inference on his small GPU cluster?

---

### Q7  
Vimarsh recently came across what is known as **"batching"** prompts together.  
He decides to save compute and memory resources, and sets the batch size to **64**.  
This approach works well, as a lot of compute power is saved due to GPUs working less frequently, and in parallel.  
However, users complain of long waiting time after giving a prompt to DaSHGPT.  

Why has this happened?  
What would you do to fix this?

---

### Q8 – Q10  
Based on the following case study:

Atharva has recently read the paper on attention.  
Thinking that *"Attention is all He Needs"*, he begins working on his own transformer/LLM.  
He's able to code up almost all aspects of the neural networks, but needs help in some other ones.  

---

### Q8  
Assume that you have a sentence that is of the form:  
> "The quick brown fox jumps over the lazy ___"

You want the LLM to predict the next token in the sequence.  
Atharva's LLM does this by tokenizing the input sentence (assume one token = one word in the sentence),  
pushing it into the transformer's attention layers, computes all the attention dot products for the sentence,  
then multiplies it by the value matrix, finds the linear combination and updates the embedded vector,  
and finally passes it through the FFNN, repeating this procedure from the beginning each time a new token is added,  
thus getting the result using softmax regression.  

However, Atharva's LLM seems to be very slow at producing the output.  
What could be the issue?

---

### Q9  
Atharva has realized his mistake now, and is set to correct it.  
He now stores the required computations in his GPU VRAM.  
Assuming that Atharva is using only **1 GPU with 16GB VRAM**, a **7B parameter model** with **FP16 precision** consumes about **14 GB** of his VRAM.  
Happy that his LLM fits in the GPU, Atharva begins inference, but soon realises that the time taken to produce an output keeps increasing as he gives more and more prompts, eventually piling up to an unacceptable delay.  

Why has this happened?  
What is the fix that comes to your mind?  
Assume that the prompts are interdependent (i.e. prompt 1 uses context from prompt 2, and so on).  

*(Hint: Think in terms of where and how the pre-computed values are being stored and accessed. You ideally do not want the values to consume GPU VRAM.)*

---

### Q10  
Great, Atharva has listened to your advice and implemented "the fix" you've mentioned above.  
However, instead of an eventual, unacceptable delay, Atharva now has a fixed but noticeable delay in producing an output for each token of every prompt.  

Why has this happened?  
(Hint: think about where the KV cache is stored now, and what happens each time a new token is generated. Can this data movement be reduced?)  
Can you think of a simple way to reduce this delay?

---

### (BONUS) Q11  
Congratulations on making it this far into the assignment :)  

To give you a small taste on what the **LLM inference scheduling** project is working on, here's a toy problem for you:  
*(Do note that this is a very advanced problem for this assignment).*

---

You have a **4-node setup:**

- **C1, C2, C3** → 1 GPU each, **8 GB VRAM**  
- **C4** → 2 GPUs (**A1, A2**), **12 GB VRAM each**  

The network between nodes is **slow** (considerable communication delay).  
**NVLink** is present within C4 (transfer between **A1 ↔ A2** is fast).  
You have a **3B parameter model in FP16** (6 GB VRAM per model instance).

---

#### Your task:
Design a scheduling algorithm to assign prompts to GPUs based on their total length (**input + output**), classified as **short / medium / large**.  
(Assume output length is known beforehand — a very unrealistic one, but fits for this case.)  

You may draw a graph or diagram to explain your approach.  
Proper justification **MUST** be provided for your approach.

**Hints:**
- Consider VRAM constraints: a GPU must fit the model + KV cache for the assigned prompt.  
- Prefer fast local connections for prompts that require multiple GPUs.  
- Assign short prompts to smaller GPUs, and long prompts to larger GPUs.  
- Aim for load balancing while minimizing communication overhead.  

If you're able to figure out an optimal scheduling algorithm for the above case, list **two problems** you'd face if you were to drop the unrealistic assumption made above.

## Final note
Congratulations on finishing the Level 1 of the SysML assignment :D Way to go!
I hope this assignment helped you learn more about the working of LLMs, and was able to give you a gist of what sort of things we work on over at the lab.

I'd love to hear some feedback from you, and it would be appreciated if you could share your experience of navigating through the assignment in given Google form.

Thank you and All the Best for the next levels! Hope to see you at DaSHLab :)
