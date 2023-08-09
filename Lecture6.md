# Lecture 6

## Finding Research Topics

- two basic starting points:
  - start with a (domain) problem of interest and try to find good/better ways to address it than are currently known/used
  - start with a technical method/approach of interest, and work out good ways to extend or improve it or new ways to apply it

- project types
  - find an application/task of interest and explore how to approach/solve it effectively, often with an existing model
  - implement a complex neural architecture and demonstrate its performance on some data
  - come up with a new or variant neural network model and explore its empirical success
  - analyze the behavior of a model: how it represents linguistic knowledge or what kinds of phenomena it can handle or errors that it makes
  - show some interesting, non-trivial properties of a model type, data, or a data representation (theoretical project)

## Finding an interesting place to start

- useful websites
  - ACL anthology for NLP papers: <https://www.aclweb/prg/anthology>
  - Major ML conferences: [NeurIPS](https://papers.nips.cc), ICML, [ICLR](https://openreview.net/group?id=ICLR.cc)
  - <http://web.stanford.edu/class/cs224n/>
  - <https://arxiv.org>
  - <http://www.arxiv-sanity.com>
  - info on the state-of-the-art (not always correct):
    - <https://paperswithcode.com/sota>
    - <https://nlpprogress.com>
    - <https://gluebenchmark.com/leaderboard>
    - <https://www.conll.org/previous-tasks>

- NLP in 2023:
  - pip install transformers # By Huggingface
  - fine-tuning
  - particular tasks
  - ...

- exciting areas in 2023
  - evaluating and improving models for something other than accuracy
    - adaptation when there is domain shift (train data from one area and test data from another area)
    - evaluating the robustness of models in general: <https://robustnessgym.com>
  - doing empirical work looking at what large pre-trained models have learned
  - working out how to get knowledge and good task performance from large models for particular tasks without much data (transfer learning, etc.)
  - looking at the bias, trustworthiness, and explainability of large models
  - working on how to augment the data for models to improve performance
  - looking at low resource language or problems
  - improving performance on the tail of rare stuff, addressing bias
  - scaling models up and down
    - building big models is BIG
    - building small, performant models is also BIG
      - [model pruning](https://papers.nips.cc/paper/2020/file/eae15aabaa768ae4a5993a8a4f4fa6e4-Paper.pdf)
      - [model quantization](https://arxiv.org/pdf/2004.07320.pdf)
      - how well can you do QA in 6GB or 500MB: <https://efficientqa.github.io/>
  - looking to achieve more advanced functionalities
    - compositionality, systematic generalization, fast learning(meta-learning) on smaller problems and amounts of data, and more quickly
    - [COGS](https://github.com/najoungkim/COGS)
    - [gSCAN](https://arxiv.org/abs/2003.05161)

## Finding data

- <https://catalog.ldc.upenn.edu/>
- [Huggingface](https://huggingface.co/datasets)
- [Paperwithcode](https://www.paperswithcode.com/datasets?mod=texts&page=1)
- Machine translation: <http://statmt.org/>
- Dependency parsing: <https://universaldependencies.org/>
- lists of datasets
  - <https://machinelearningmastery.com/datasets-natural-language-processing/>
  - <https://github.com/niderhoff/nlp-datasets>
- particular things
  - <https://gluebenchmark.com/tasks>
  - <https://nlp.stanford.edu/sentiment/>
  - <https://research.fb.com/downloads/babi/>

## Pots of data

- keep the test data until the end
- create a dev set/tune set by splitting the training data
- **Cross-validation** is a technique for maximizing data when you don't have much
- training(these sets need to be completely distinct)
  - build a model on a _training set_
  - set further hyperparameters on another, independent set of data, the _tuning set_ (the _training set_ for the hyperparameters)
  - measure progress on a _dev set_(development test set or validation set)
  - only at the end, you evaluate and present final numbers on a _test set_
- for good generalization performance: regularize your model until it doesn't overfit on _dev data_

## An example

1. Define Task
   - example: Summarization
2. Define Dataset
   1. Search for academic datasets
        - newsroom summarization dataset: <http://lil.nlp.cornell.edu/newsroom/>
   2. Define your own data
        - be creative, like generating advertising tweet from a news story
3. Dataset hygiene
   - right at the beginning, separate off dev data and test data splits
4. Define your metric(s)
   - search online for well established metrics on this task
   - summarization: Rouge(Recall-Oriented Understudy for Gisting Evaluation) which defines _n_-gram overlap to human summaries
   - Human evaluation is still much better for summarization
5. Establish a baseline
   - implement the simplest model first(like logistic regression on unigrams and bigrams or averaging word vectors)
     - for summarization: LEAD-3 baseline
   - compute metrics on train and dev not dest
   - analyze errors
6. Implement existing neural net model
   - compute metric on train and dev
   - analyze output and errors
7. Always be close to your data
   - visualize the dataset
   - collect summary statistics
   - look at errors
   - analyze how different hyperparameters affect performance
8. Try out different models and model variants & Aim to iterate quickly via having a good experimental setup
   - fixed window neural model
   - recurrent neural network
   - recursive neural network
   - convolutional neural network
   - attention-based model/transformer
   - ...
