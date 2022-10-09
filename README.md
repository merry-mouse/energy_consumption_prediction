# Energy consumption prediction model

Energy Consumption (in Sweden) Prediction Using Python & Machine Learning (LSTM). This is a program that shows how to build an artificial neural network called Long Short Term Memory to predict the future consumtion of energy (load in Megawatts).

# libraries and modules

There are many modules and libraries that has to be imported and installed , most importantly pandas, numpy, tensorflow, scikit-learn and others. Can be installed with pip.

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install seaborn
pip install plotly==5.10.0
pip install tensorflow
```

```
# for mathematical computation

import numpy as np
import pandas as pd
import scipy.stats as stats

# for data visualization

import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from matplotlib.pyplot import figure
%matplotlib inline

# for creatung neural network

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import math
```

# Data

All data was downloaded from Kaggle: https://www.kaggle.com/datasets/francoisraucent/western-europe-power-consumption
For this project I decided to use dataset for Sweden. It can be found in energy_consumption_EU folder. There are several files, each one is a csv file corresponding to a different European country. Data consists of columns named "start", "end" and "load". The first two are timestamps and the last is a load in Megawatts (MW).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
