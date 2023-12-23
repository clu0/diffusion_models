# diffusion_models
Learning about diffusion models

## Simplest diffusion model

### Training
Training for a diffusion models turns out to be embaressingly simple after the simplifications done in [Ho et al. 20](https://arxiv.org/pdf/2006.11239.pdf) (but the explanation in Appendix B of [DN21](https://arxiv.org/pdf/2006.11239.pdf) is better). You simply minimize the following objective:
$$
L = \mathbb{E}[\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2]\, ,
$$
where the expectation is taken over $t \sim [1, T]$, $\epsilon \sim \mathcal N (0, I)$, $x_0 \sim q$ (the original data distribution), and $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1 - \alpha_t} \epsilon$.

And that's it. The model is usually taken to be a UNet, and the time schedule details can be found in Section 4 of Ho et al. 20:
- Total time steps $T = 1000$
- $\alpha_t = \prod_{i = 1}^t (1 - \beta_i)$
- $\beta$ ranges linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.

That seems to be it! We will implement it in `train.py`.