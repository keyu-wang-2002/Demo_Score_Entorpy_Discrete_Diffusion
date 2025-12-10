# Score Entropy Discrete Diffusion

This repo contains some simple demos for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834). 

## Installation

Simply run
```
pip install -r requirements.txt
```
Or you can directly run these .ipynb files in Google Colab, without installing any packages.


## Compare Score Entropy Loss with MSE Loss in Concrete Score Matching
Concrete Score Matching (CSM) learns discrete scores using  
\[
\left( s_\theta(x_t, t)_y - \frac{p_t(y)}{p_t(x)} \right)^2 .
\]
As noted in Score Entropy Discrete Diffusion (SEDD), the ℓ2 objective is incompatible with the requirement that \(\frac{p_t(y)}{p_t(x)} > 0\). It does not strongly penalize zero or negative predictions, often causing unstable or divergent training. As a result, CSM, despite its theoretical elegance, tends to struggle in practice.


![Gradient Comparison](figure/sedd_vs_csm.png)
