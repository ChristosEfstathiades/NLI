---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/ChristosEfstathiades/NLI

---

# Model Card for d86617ce-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to 
    detect whether a hypothesis is true based on the given premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model uses a linear support vector classifier (LSVC) 
    with BERT as a feature extractor.

- **Developed by:** Christos Efstathiades and Benjamin Eichhoefer
- **Language(s):** English
- **Model type:** LSVC
- **Model architecture:** NA
- **Finetuned from model [optional]:** SVM

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://scikit-learn.org/stable/api/sklearn.svm.html
- **Paper or documentation:** https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K premise-hypothesis pairs.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - max_iter: 1000
      - dual: False
      - C: 1.0
      - penalty: L2
      - loss: squared_hinge

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 9 minutes
      - model size: 50KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Development set provided, amounting to 6.7K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 68% and an accuracy of 68%.

## Technical Specifications

### Hardware


      - RAM: at least 15 GB
      - Storage: at least 2GB,
      - GPU: T4

### Software


      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113
      - pandas
      - sklearn
      - joblib
      - numpy

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any premise or hypothesis longer than
      42 words will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
