# Multi-Task Deep Morph Analyzer  [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

A multi-task learning CNN-RNN model combined together with the potential of task-optimized phonetic features to predict the Lemma, POS category, Gender, Number, Person, Case, and Tense-aspect-mood (TAM) of Hindi words. 

![image](https://github.com/Saurav0074/morph_analyzer/blob/master/src/images/sample.png)

## Framework

![image2](https://github.com/Saurav0074/morph_analyzer/blob/master/src/images/morph_analyzer_model.png)

# Getting started

### Clone the repository

```
git clone git@github.com:Saurav0074/morph_analyzer.git
cd morph_analyzer
```

### Provide the arguments

The file `main.py` takes the following command-line arguments: 

| Argument | Values | Required | Specification |
| ------- | ------- | ------------- | ------------ |
| lang     | hindi, urdu  | Yes | Language |
| mode     | train, test and predict (i.e., no gold labels required). | Yes |  Training, testing and predictions. |
| phonetic | True/1/yes/y/t and False/0/no/n/f. | No (default=`False`) | Use MOO-driven phonological features or not. |
| freezing | "       "      and "       " | No (default=`False`) | Use the [FreezeOut](https://arxiv.org/abs/1706.04983) training strategy or not. |

`train` and `test` modes operate upon the standard train-test split specified by the HDTB and UDTB datasets (see `datasets` [README](https://github.com/Saurav0074/morph_analyzer/blob/master/datasets/README.md) while `predict` uses the text provided manually in `src/[lang]_predict_data/`.

#### Sample run commands: 

```python
>>> python main.py --lang urdu --mode train --phonetic true --freezing true #train

>>> python main.py --lang urdu --mode test --phonetic true --freezing true #test

>>> python main.py --lang urdu --mode predict --phonetic true --freezing true #predict
```

For prediction, the plain text should be provided within `src/[lang]_predict_data/test_data.txt`.

### Outputs

For the test mode:

- the predicted roots and features as well as their gold-labelled counterparts are written to separate files within `output/[lang]/roots.txt, feature_0.txt, ..., feature_6.txt`.
- Micro-averaged precision-recall graphs are stored in `graph_outputs/[lang]/`.

For the predict mode, all the predictions (i.e., roots + features) are written to - `output/[lang]/predictions.txt`.

### Graph outputs for multi-objective optimization of phonological features

- Cubic-spline interpolations depicting validation accuracies ordered by population:

![moo](https://github.com/Saurav0074/morph_analyzer/blob/master/src/images/cubic-splines.png)


# Citation

If this repo was helpful in your research, consider citing our work:

```
@article{jha2018multi,
  title={Multi Task Deep Morphological Analyzer: Context Aware Joint Morphological Tagging and Lemma Prediction},
  author={Jha, Saurav and Sudhakar, Akhilesh and Singh, Anil Kumar},
  journal={arXiv preprint arXiv:1811.08619},
  year={2018}
}
```
