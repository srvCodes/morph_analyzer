# Hindi Morph Analyzer  [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Getting started

### Clone the repository

```
git clone git@github.com:Saurav0074/morph_analyzer.git
cd morph_analyzer
```

### Provide the arguments

The file `main.py` takes two command-line arguments: (a) --lang with possible values `hindi` and `urdu`, and (b) --mode with values `train` for training, `test` for testing and evaluating, and `predict` for prediction. Train and test modes perform upon the standard train-test split specified by the HDTB and UDTB datasets.

```python
>>> python main.py --lang hindi --mode train
```

For prediction, the test sentences should be provided within `src/hindi/test_data.txt`.

### Outputs

For the test mode, the gold and predicted roots as well as features are all written to separate files within `output/roots.txt, feature_0.txt, ..., feature_6.txt`.

For the predict mode, all the predictions (i.e., roots + features) get written to the same file - `output/predictions.txt`.


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
