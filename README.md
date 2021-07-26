## Nearest Instance Counterfactual Explanations (NICE)

NICE is an algorithm to generate Counterfactual Explanations for heterogeneous tabular data. Our approach exploits 
information from a nearest instance to speed up the search process and guarantee that an explanation will be found.

### Installation

Install NICE through Pypi

```bash
pip install NICEx
```

or github

```bash
pip install git git+https://github.com/ADMantwerp/nice.git 
```

### Usage

NICE requires acces to the prediction score and trainingdata to generate counterfactual explanations.
```python
from nice.explainers import NICE

# Initialize NICE by specifing the optimization strategy
NICE_explainer = NICE(optimization='sparsity')
# Fit our NICE explainer on the training data and classifier
NICE_explainer.fit(predict_fn,X_train,cat_feat,num_feat,y_train)
# explain an instance
NICE_explainer.explain(x)
```

### Examples
 [NICE on Adult](https://github.com/DBrughmans/NICE/blob/master/examples/NICE_adult.ipynb)
 
### References
[NICE: NICE: An Algorithm for Nearest Instance Counterfactual Explanations](https://arxiv.org/abs/2104.07411)
