# Alpha Research with AI
## Project Description
Please refer to the medium post: https://yuki678.medium.com/how-to-build-quant-algorithmic-trading-model-in-python-12abab49abe3?sk=56d5b2b038ce6aefa6c2049cff9e89b6 

## File Description
### Alpha Research Base.ipynb
Jupyter notebook to execute alpha research process without Machine Learning.
Functions are defined in `myalpha` package.

### Alpha Research AI.ipynb
Add Machine Learning to generate Alpha vector on top of the base process.

### myalpha package
This include the following modules used in the notebook for basic alpha research

#### datautil.py
Utility functions to handle input data

#### riskmodel.py
Statistical (PCA) based risk model functions

#### alphas.py
Defines Alpha Factors

#### mlutil.py
Classes and functions for Machine Learning process to create AI Alpha

#### performance.py
Performance measurement and visualization functions using `alphalens`

#### optimization.py
Optimization classes to calculate optimal holding based on the constraints
