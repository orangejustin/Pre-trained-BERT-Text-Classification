# PA 4: Pre-trained BERT Text Classification Application with Amazon-Massive-Intent Dataset

This respository is refered and modified from this repository of the [paper](https://arxiv.org/abs/2109.03079).

## Contributors
Wenqian, Keyu, Xiaotong, Yunze, Zecheng

## Task

The goal of this project is to generate user intent from the amazon massive intent
dataset. The inputs are different texts that are assigned to different labels based
on the text’s intentions. We first put all the inputs into an encoder of pre trained
transformer BERT, then we take the token of the last hidden state as the output
from the encoder to a dropout layer with a specific dropout rate, and last we put the
dropout layer into the classifier to obtain the predictions.


## How to run
We completed the implementation of the pre-trained BERT 
function tailored by fined-turned and trained by different
loss functions in model.py. The implementation details of the
SupCon loss function can be seen in loss.py, with the main 
ideas and code from https://arxiv.org/pdf/2004.11362.pdf. 

Just run the code we provided in run.sh in the terminal and
you will be able to achieve the results of the corresponding model.

You will get the best baseline model performance results by running:
```
python main.py --ignore-cache --n-epochs 10 --max-len 64 --task baseline --drop-rate 0.2 --hidden-dim 4096 --learning-rate 0.0005
```
You will get the best custom model performance results by running:
```
python main.py --ignore-cache --n-epochs 10 --task custom --drop-rate 0.1 --learning-rate 0.001  --hidden-dim 4096 --warm-up-step 200 --scheduler cosine
```
Experiments for comparative learning of different loss functions.
- SupContrastive
```
python main.py --ignore-cache --drop-rate 0.2 --learning-rate 0.0005 --max-len 64 --hidden-dim 16384 --batch-size 256 --task supcon
```
- SimCLR
```
0.2 --learning-rate 0.0005 --max-len 64 --hidden-dim 16384 --batch-size 256 --task simclr
```
## Files

- `main.py`: Main driver class
- `dataloader.py`: data loader file which mainly preprocess the text data in prepare_features function 
- `load.py`: load the tokenizer in the load_tokenizer function which you need to write
- `argument.py`: Arguments provided for parameter tuning or experimental loading 
- `model.py`: Provides the codes for the model used in the experiemnt
- `loss.py`: SupContrast and SimCLR losses from https://github.com/HobbitLong/SupContrast/tree/master 
- `run.sh`: Runs your experiment here
- `results`: Stores your experiment results 