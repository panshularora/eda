# Hyperparameter sweep

Grouped 5-fold CV macro-F1 on the **training split only**; the held-out
slice took no part in any of this. Produced by `python round2/src/sweep.py`.

## sentiment

Best regularisation **C = 0.1** (0.6227); best character range **2-5** (0.6131).

### Regularisation (word + char + surface, LinearSVC)

| C | CV macro-F1 | SD |
|---|---|---|
| 0.1 | 0.6227 | 0.012 |
| 0.25 | 0.6208 | 0.011 |
| 0.5 | 0.6125 | 0.014 |
| 1.0 | 0.6008 | 0.015 |
| 2.0 | 0.5916 | 0.013 |
| 4.0 | 0.5872 | 0.016 |
| 8.0 | 0.5838 | 0.014 |
| 16.0 | 0.5818 | 0.013 |
| 32.0 | 0.5814 | 0.013 |

### Character n-gram range (character block alone)

| range | CV macro-F1 | SD |
|---|---|---|
| 2-2 | 0.5555 | 0.015 |
| 2-3 | 0.5961 | 0.005 |
| 2-4 | 0.6092 | 0.014 |
| 2-5 | 0.6131 | 0.016 |
| 3-5 | 0.6055 | 0.016 |
| 3-6 | 0.6060 | 0.017 |

## topic

Best regularisation **C = 32.0** (0.7287); best character range **2-3** (0.9140).

### Regularisation (word + char + surface, LinearSVC)

| C | CV macro-F1 | SD |
|---|---|---|
| 0.1 | 0.5986 | 0.029 |
| 0.25 | 0.6499 | 0.043 |
| 0.5 | 0.6892 | 0.041 |
| 1.0 | 0.7105 | 0.038 |
| 2.0 | 0.7160 | 0.038 |
| 4.0 | 0.7232 | 0.041 |
| 8.0 | 0.7245 | 0.042 |
| 16.0 | 0.7276 | 0.040 |
| 32.0 | 0.7287 | 0.041 |

### Character n-gram range (character block alone)

| range | CV macro-F1 | SD |
|---|---|---|
| 2-2 | 0.6746 | 0.021 |
| 2-3 | 0.9140 | 0.020 |
| 2-4 | 0.9057 | 0.017 |
| 2-5 | 0.8746 | 0.024 |
| 3-5 | 0.8508 | 0.033 |
| 3-6 | 0.8223 | 0.032 |

## Reading

The two tasks land in opposite regimes. Sentiment wants heavy
regularisation over long character n-grams: it is noisy, semantic, and the
model has to generalise. Topic wants light regularisation over 2-3 character
n-grams - which is exactly the length of the trigger substrings (`ui`, `ban`,
`app`, `bug`) that `audit_labels.py` recovered from the labels. The
hyperparameter that wins is telling you what the label is made of.
