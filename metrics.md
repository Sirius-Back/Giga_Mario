# metrics.md

## Overview

This project uses **TorchMetrics** for all validation metrics.

Why:

- battle-tested implementation;
- distributed training support (DDP);
- accumulates values across batches automatically;
- identical behaviour on CPU/GPU;
- easy logging each epoch.

Documentation

https://lightning.ai/docs/torchmetrics/

GitHub

https://github.com/Lightning-AI/torchmetrics

---

# 1. Validation Loss

## Definition

The primary optimization objective evaluated on the validation dataset.

Usually one of:

- MSELoss
- HuberLoss
- PoissonNLLLoss

depending on prediction target.

This metric is used for:

- early stopping
- model checkpoint selection

Example

```python
criterion = torch.nn.MSELoss()

...

val_loss = criterion(pred, target)
```

Logged every validation epoch.

---

# 2. Pearson Correlation

## Purpose

Measures linear agreement between predicted and true expression values.

Range

-1 ... 1

Higher is better.

The main metric reported in most transcript prediction papers.

TorchMetrics

```python
from torchmetrics.regression import PearsonCorrCoef

metric = PearsonCorrCoef()
```

Usage

```python
metric.update(pred, target)

...

pearson = metric.compute()

metric.reset()
```

MetricCollection

```python
from torchmetrics import MetricCollection
from torchmetrics.regression import PearsonCorrCoef

metrics = MetricCollection({
    "pearson": PearsonCorrCoef()
})
```

Documentation

https://lightning.ai/docs/torchmetrics/stable/regression/pearson_corr_coef.html

---

# 3. Spearman Correlation

## Purpose

Measures agreement of gene rankings.

Unlike Pearson, it is robust to monotonic nonlinear relationships and outliers.

Range

-1 ... 1

Higher is better.

Very common for genomics.

TorchMetrics

```python
from torchmetrics.regression import SpearmanCorrCoef

metric = SpearmanCorrCoef()
```

Usage

```python
metric.update(pred, target)

...

rho = metric.compute()

metric.reset()
```

Documentation

https://lightning.ai/docs/torchmetrics/stable/regression/spearman_corr_coef.html

---

# 4. Mean Squared Error (MSE)

## Definition

Average squared prediction error.

Lower is better.

Formula

MSE = mean((prediction - target)^2)

TorchMetrics

```python
from torchmetrics.regression import MeanSquaredError

metric = MeanSquaredError()
```

Usage

```python
metric.update(pred, target)

mse = metric.compute()
```

Documentation

https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html

---

# 5. Root Mean Squared Error (RMSE)

## Definition

Square root of Mean Squared Error.

Has the same units as the prediction target.

Lower is better.

TorchMetrics

```python
from torchmetrics.regression import MeanSquaredError

metric = MeanSquaredError(squared=False)
```

Usage

```python
metric.update(pred, target)

rmse = metric.compute()
```

Documentation

https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html

---

# 6. Mean Absolute Error (MAE)

## Definition

Average absolute prediction error.

Less sensitive to outliers than MSE.

Lower is better.

Formula

MAE = mean(abs(prediction - target))

TorchMetrics

```python
from torchmetrics.regression import MeanAbsoluteError

metric = MeanAbsoluteError()
```

Usage

```python
metric.update(pred, target)

mae = metric.compute()
```

Documentation

https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html

---

# Recommended MetricCollection

```python
from torchmetrics import MetricCollection

from torchmetrics.regression import (
    PearsonCorrCoef,
    SpearmanCorrCoef,
    MeanSquaredError,
    MeanAbsoluteError,
)

metrics = MetricCollection({
    "pearson": PearsonCorrCoef(),
    "spearman": SpearmanCorrCoef(),
    "mse": MeanSquaredError(),
    "rmse": MeanSquaredError(squared=False),
    "mae": MeanAbsoluteError(),
})
```

Validation loop

```python
metrics.reset()

for batch in val_loader:

    pred = model(batch)

    metrics.update(pred, target)

epoch_metrics = metrics.compute()

for name, value in epoch_metrics.items():

    logger.log(name, value)
```

The metrics should be computed **once per validation epoch**, after processing the entire validation dataset.

Do **NOT** average batch-wise metric values manually.
TorchMetrics accumulates sufficient statistics internally and computes the correct epoch-level metric.

---

# 7. R² Score (Coefficient of Determination)

## Purpose

Measures the fraction of variance in the target explained by the model.

Unlike Pearson correlation, R² penalizes systematic bias in the predictions.

Range

(-∞, 1]

Interpretation

- 1.0 = perfect prediction
- 0.0 = no better than predicting the mean
- <0 = worse than predicting the mean

Higher is better.

TorchMetrics

```python
from torchmetrics.regression import R2Score

metric = R2Score()
```

Usage

```python
metric.update(pred, target)

r2 = metric.compute()

metric.reset()
```

MetricCollection

```python
from torchmetrics.regression import R2Score

metrics = MetricCollection({
    "r2": R2Score()
})
```

Documentation

https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html

GitHub

https://github.com/Lightning-AI/torchmetrics

---

# 8. Median Gene-wise Pearson Correlation

## Purpose

Measures prediction quality independently for each gene.

Instead of computing one global Pearson coefficient across all values, compute one Pearson correlation per gene across all validation samples.

The final metric is

median(
    Pearson(gene1),
    Pearson(gene2),
    ...
)

Why median?

- robust to noisy genes
- prevents a few highly expressed genes from dominating the score
- common in transcriptomics benchmarks

There is currently **no official TorchMetrics implementation**.

Recommended implementation

```python
import torch

def genewise_pearson(pred, target):
    """
    pred   : (N_samples, N_genes)
    target : (N_samples, N_genes)
    """

    pred = pred - pred.mean(dim=0)
    target = target - target.mean(dim=0)

    corr = (
        (pred * target).sum(dim=0)
        /
        (
            torch.sqrt((pred ** 2).sum(dim=0))
            * torch.sqrt((target ** 2).sum(dim=0))
            + 1e-8
        )
    )

    return corr
```

Epoch metric

```python
corr = genewise_pearson(pred, target)

median_genewise_pearson = corr.median()
```

Recommended logging

```python
logger.log(
    "genewise_pearson_median",
    median_genewise_pearson
)
```

---

# 9. Median Sample-wise Pearson Correlation

## Purpose

Measures whether the predicted expression profile of each biological sample matches the real profile.

Instead of correlating genes,

compute Pearson separately for every sample.

The final metric is

median(
    Pearson(sample1),
    Pearson(sample2),
    ...
)

Useful because

- evaluates reconstruction of complete transcriptomes
- insensitive to a few difficult samples
- frequently reported in foundation-model benchmarks

Implementation

```python
import torch

def samplewise_pearson(pred, target):
    """
    pred   : (N_samples, N_genes)
    target : (N_samples, N_genes)
    """

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    corr = (
        (pred * target).sum(dim=1)
        /
        (
            torch.sqrt((pred ** 2).sum(dim=1))
            * torch.sqrt((target ** 2).sum(dim=1))
            + 1e-8
        )
    )

    return corr
```

Epoch metric

```python
corr = samplewise_pearson(pred, target)

median_samplewise_pearson = corr.median()
```

Recommended logging

```python
logger.log(
    "samplewise_pearson_median",
    median_samplewise_pearson
)
```

---

# Recommended validation outputs

Every validation epoch should log

```text
val_loss
pearson
spearman
mse
rmse
mae
r2
genewise_pearson_median
samplewise_pearson_median
```

---

# Validation loop example

```python
metrics.reset()

all_pred = []
all_target = []

for batch in val_loader:

    pred = model(batch)

    metrics.update(pred, target)

    all_pred.append(pred.detach().cpu())
    all_target.append(target.detach().cpu())

epoch_metrics = metrics.compute()

pred = torch.cat(all_pred)
target = torch.cat(all_target)

epoch_metrics["genewise_pearson_median"] = (
    genewise_pearson(pred, target).median()
)

epoch_metrics["samplewise_pearson_median"] = (
    samplewise_pearson(pred, target).median()
)

for k, v in epoch_metrics.items():
    logger.log(k, float(v))
```

This approach ensures:

- all standard metrics are computed with TorchMetrics;
- custom transcriptomics metrics are computed only once per validation epoch;
- metrics remain deterministic and compatible with distributed training.