# Score Entropy Discrete Diffusion

This repo contains some simple demos for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834), supporting Keyu Wang's presentation for SEDD at the seminar ` Machine Learning Methods for Scientific Discovery` at the University of Tuebingen in 25ws.

## Installation

Simply run
```
pip install -r requirements.txt
```
Or you can directly run these .ipynb files in Google Colab, without installing any packages.


## Compare Score Entropy Loss with MSE Loss in Concrete Score Matching (`sedd_vs_csm.ipynb`)
Concrete Score Matching (CSM) learns discrete scores using  $( s_\theta(x_t, t)_y - \frac{p_t(y)}{p_t(x)} )^2$. 

As noted in Score Entropy Discrete Diffusion (SEDD), the $\ell_2$ objective is incompatible with the requirement that $p_t(y)/p_t(x) > 0$. It does not strongly penalize zero or negative predictions, often causing unstable or divergent training. Thus, despite its theoretical appeal, CSM tends to struggle in practice. To address these limitations, SEDD introduces the **Score Entropy Loss** as a replacement for the MSE objective:  $s - \frac{p_t(y)}{p_t(x)}\log(s)$ where $s = s_\theta(x_t, t)_y$.

In `sedd_vs_csm.ipynb`, we compare the two loss functions side by side, and the results are shown below.  
The top row corresponds to CSM and the bottom row to SEDD.  
The first column shows the target true ratio, while columns 2–7 display the learned ratios at steps 0, 20, 40, 60, 80, and 100, respectively.

![Gradient Comparison](figure/sedd_vs_csm.png)

We can see that SEDD converges to the true ratio more efficiently and more robustly.


## Demo for SEDD I: One-Step Generation from One-Step Noise (`SEDD_miniDemo_1.ipynb`)

After applying a single noise step to an original sentence, we find that a network trained with SEDD is able to reliably reconstruct a coherent sentence

Training-Time Denoising Log
| Clean Sentence      | Noised Input            | Prediction        |
|----------------------|--------------------------|--------------------|
| We love NLP .        | We AI love You          | We love NLP .      |
| We love NLP .        | We love You .           | We love NLP .      |
| You love NLP .       | You . NLP .             | You love NLP .     |
| I love AI .          | I love AI You           | I love AI .        |
| You love NLP .       | <pad> love AI love      | I love AI .        |
| I love NLP .         | We <pad> I .            | We love NLP .      |
| I love AI .          | We love <pad> We        | We love AI .       |
| I love AI .          | . love love I           | We love AI .       |
| I love AI .          | love AI You .           | I love NLP .       |
| You love AI .        | love NLP AI love        | We love AI .       |


## Demo for SEDD II: Generation from Fully Masked Inputs (`SEDD_miniDemo_2.ipynb`)

To further test the robustness of SEDD on discrete text, we run an extreme setting where the corruption process replaces **all tokens** with the `[MASK]` symbol (i.e., full information removal).  
The vocabulary size is 9, with `MASK_ID = 8`.

Despite receiving **no lexical information** from the input, the model learns to produce fluent, grammatical outputs that match the training distribution.

Training-Time Denoising Log (Mask Noise)
| Clean Sentence      | Noised Input                     | Prediction          |
|----------------------|----------------------------------|----------------------|
| We love NLP .        | [MASK] [MASK] [MASK] [MASK]      | I love NLP .         |
| I love AI .          | [MASK] [MASK] [MASK] [MASK]      | We love NLP .        |
| I love AI .          | [MASK] [MASK] [MASK] [MASK]      | You love NLP .       |
| We love NLP .        | We love [MASK] [MASK]            | We love NLP .        |
| We love AI .         | [MASK] [MASK] [MASK] .           | You love AI .        |
| You love AI .        | [MASK] [MASK] [MASK] [MASK]      | We love AI .         |
| You love AI .        | [MASK] [MASK] [MASK] [MASK]      | You love AI .        |
| You love AI .        | You [MASK] AI [MASK]             | You love AI .        |
| We love NLP .        | [MASK] [MASK] [MASK] [MASK]      | You love AI .        |
| I love AI .          | [MASK] love AI [MASK]            | I love AI .          |

FULL-MASK Generation
| Input x_t                      | Prediction        |
|--------------------------------|--------------------|
| [MASK] [MASK] [MASK] [MASK]    | I love NLP .       |
| [MASK] [MASK] [MASK] [MASK]    | We love NLP .      |
| [MASK] [MASK] [MASK] [MASK]    | You love NLP .     |
| [MASK] [MASK] [MASK] [MASK]    | I love NLP .       |
| [MASK] [MASK] [MASK] [MASK]    | You love AI .      |
| [MASK] [MASK] [MASK] [MASK]    | We love AI .       |
| [MASK] [MASK] [MASK] [MASK]    | You love AI .      |
| [MASK] [MASK] [MASK] [MASK]    | I love AI .        |
| [MASK] [MASK] [MASK] [MASK]    | You love AI .      |
| [MASK] [MASK] [MASK] [MASK]    | I love NLP .       |


## Other Recommend Resources

The original paper is not easy to follow, so I high recommend to read the authors' blog first:
[SEDD Blog](https://aaronlou.com/blog/2024/discrete-diffusion/)

Another simple implementation of SEDD (but more complex than this repo cause this repo maybe the simplest):
[SEDD Simple Implementation](https://github.com/emadyagh/Simple-Implementation-of-Score-Entropy-Discrete-Diffusion-SEDD-for-Text-Generation)

