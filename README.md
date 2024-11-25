# MVIFSA: Enhancing Relation Detection in Knowledge Base Question Answering through Multi-View Information Fusion and Self-Attention

#### The framework of our MVIFSA:

MVIFSA comprises five network layers: a multi-view embedding layer, an information fusion layer, a complex information representation layer, a residual learning layer, and a self-attention layer. The model diagram is as follows.

![](C:\Users\27663\Desktop\论文\MVIFSA投EAAI\TODO\MVIFSA\img\MVIFSA.png)

#### How to run our code?

#### Preliminary

You can download the word encoding files required for the experiment from  [GloVe](https://nlp.stanford.edu/projects/glove/).

The environment required for the code is in requirements.

#### Data preprocessing

1. Configure the dataset in `config.ini` and the maximum length.

2. You can run `preprocess.py` to preprocess the dataset to get various training and test sets for model training.

Train model.

#### Model train

The `MVIFSA.py` file contains the model building and training code. After data preprocessing, you can directly run the file to get the model training results.

#### Model evaluate

Once you have saved your training model, run the `MVIFSA_eval.py` file, which allows you to evaluate the trained model.

### *Thank you for your interest in our work, and feel free to contact the author with any questions you may have.*
