# torch-lr-scheduler [![Build Status](https://travis-ci.com/FebruaryBreeze/torch-lr-scheduler.svg?branch=master)](https://travis-ci.com/FebruaryBreeze/torch-lr-scheduler) [![codecov](https://codecov.io/gh/FebruaryBreeze/torch-lr-scheduler/branch/master/graph/badge.svg)](https://codecov.io/gh/FebruaryBreeze/torch-lr-scheduler) [![PyPI version](https://badge.fury.io/py/torch-lr-scheduler.svg)](https://pypi.org/project/torch-lr-scheduler/)

PyTorch Optimizer Lr Scheduler.

## Installation

Need Python 3.6+.

```bash
pip install torch-lr-scheduler
```

## Usage

```python
import torch_lr_scheduler


lr_scheduler = torch_lr_scheduler.factory(config={
    'line_chain': [{
        # warm up to 0.8
        'mode': 'linear',
        'ratio': 0.01,
        'start': 0.2,
        'target': 0.8
    }, {
        # cosine to 0.0
        'mode': 'cosine',
        'ratio': 1.0,
        'target': 0.0
    }]
})

print(lr_scheduler)
#> LrScheduler (learning_rate_scale: 1.0) with LineChain (
#>     1.0%, linear from 0.2 to 0.8,
#>   100.0%, cosine from 0.8 to 0.0,
#> )
```
