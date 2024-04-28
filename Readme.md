## Assignment 3
Pre-Train Bert Model ("bert-base-uncassed") using MLM and NSP. <br> 
Then, finetuning the pre-trained models for classification and question-answering tasks.

### Files and Folders<br>
`pretrainBert.ipynb`: contains the pretraining code for Bert Model using MLM and NSP<br>
`classification.ipynb`: contains fine tuning for classification and evaluation methods<br>
`Q&A.ipynb`: contains fine tuning for the QA task<br>
`QA_evaluation.ipynb`: contains the evaluation metrics for the QA task. <br>
`documentation_NLP_3.pdf`: the documentation of the work, academic submisssion <br>
`parameter_calculation.ipynb`: contains the parameters for the pretrainedBert.ipynb, classification.ipynb, Q&A.ipynb. <br>

### Hugging Face link
[https://huggingface.co/PinkiKumari22]

### Assignment Description
Problem Statement
Pretrain an LM and fine-tune the model for a task. 

* Select the Bert-base-uncased [[code](https://huggingface.co/bert-base-uncased), [paper](https://aclanthology.org/N19-1423.pdf)] model.
* Calculate the number of parameters of the selected model from the code. Does your calculated parameters matches with the parameters reported in the respective paper. 
* Pretrain the selected model on the train split of ‘wikitext-2-raw-v1’. 
* For 5 epochs. Use the hyperparameters as per your choice. 
* Compute and report the Perplexity scores using the inbuilt function on the test split of   ‘wikitext-2-raw-v1’ for each epoch. Do scores decrease after every epoch? Why and why not?
* Push the pre-trained model to 🤗.
* Fine-tune the final pretrained model on the following three tasks:<br>
      Classification: SST-2 <br>
      Question-Answering: SQuAD <br>
* Train-test split should be 80:20, use random/stratify sampling and seed as 1. Fine-tuning should be performed on the Train split. 
* Calculate the scores for the following metrics on the test splits. Note that metrics depend on the selected task: <br>
      Classification: Accuracy, Precision, Recall, F1 <br>
* Question-Answering: squad_v2, F1, METEOR, BLEU, ROUGE, exact-match 
* Calculate the number of parameters in the model after fine-tuning. Does it remain the same as the pre-trained model? 
* Push the fine-tuned model to 🤗.
* Write appropriate comments and rationale behind:<br>
    Poor/good performance.  <br>
    Understanding from the number of parameters between pretraining and fine-tuning of the model.


### References
* https://huggingface.co/datasets/wikitext/viewer/wikitext-2-raw-v1
* https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2
* https://huggingface.co/datasets/squad_v2
