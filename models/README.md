This directory contains various deep learning models implemented in PyTorch.

To use the models, simply import them as follows:
```py
from models.transformers import iTransformer
```
or
```py
import models.transformers.iTransformer as iTransformer
```

Note your running script should be parallel to the `models` directory. For example:
```
project_dir/
├── main.py
├── models/
│   ├── __init__.py
│   ├── transformers.py
│   ├── CNN.py
│   └── ...
└── ...
```
Otherwise, you can add the `models` directory to your `PYTHONPATH` environment variable. Just add the following lines at the beginning of your `.py` or `.ipynb` file:
```py
import sys
sys.path.append('/path/to/project_dir/') # e.g. sys.path.append('D:/Programs/project_dir/')
from models.transformers import iTransformer # now you can import the models as usual
```

To be continued...