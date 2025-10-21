## Identification of LLM Bias during Venturing


### Problem
LLMs are increasingly used to assist in various text generation tasks, including such as business pitch generation for nascent entrepreneurs.
On the other hand, venture capitalists are also using LLMs to help the evaluation of multiple 


### Setup

| Name given | Generation | Evaluation |
|---|---|---|
| Case 1 | x | x |
| Case 2 | o | x |
| Case 3 | o | o |

Comparison of each case will show us the following
Case 1 - Case 2 : Bias present during generation
Case 1 - Case 3 : Bias present during evaluation
Diff-in-Diff of above : Interplay of both biases

By showing this, we aim to show that not only is the use of LLM in each step subject to the bias that LLMs entail, but also that they may even exacerbate.

### Data
1. List of names with countries and gender 
2. List of pitch ideas

### Models
- Qwen-3 
- Gemma-3
- LLama-4
- Phi-4
- Mistral-small
- GPT-oss

### Experiments

#### Case 1
Generation of pitches based on the business ideas. Will be using multiple models for this purpose.


#### Case 2
Generation of pitches with names

