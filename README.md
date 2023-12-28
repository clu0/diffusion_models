# diffusion_models
Learning about diffusion models

## Simplest diffusion model

This will be in `naive/`.
### Training
Training for a diffusion models turns out to be embaressingly simple after the simplifications done in [Ho et al. 20](https://arxiv.org/pdf/2006.11239.pdf) (but the explanation in Appendix B of [DN21](https://arxiv.org/pdf/2006.11239.pdf) is better). You simply minimize the following objective:
$$
L = \mathbb{E}[\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2]\, ,
$$
where the expectation is taken over $t \sim [1, T]$, $\epsilon \sim \mathcal N (0, I)$, $x_0 \sim q$ (the original data distribution), and $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1 - \alpha_t} \epsilon$.

And that's it. The model is usually taken to be a UNet, and the time schedule details can be found in Section 4 of Ho et al. 20:
- Total time steps $T = 1000$
- $\alpha_t = \prod_{i = 1}^t (1 - \beta_i)$ (note: our $\alpha_t$ is actually $\bar\alpha_t$ in the papers; their $\alpha_t$ is our $1 - \beta_t$).
- $\beta$ ranges linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.

That seems to be it! We will implement it in `train.py`.

## Text to image models

This will be about generating images conditional on text. We will follow the Dall-E approach by using CLIP, as in [Ramesh et al](https://arxiv.org/abs/2204.06125) (but probably more of what we're doing is from [GLIDE](https://arxiv.org/pdf/2112.10741.pdf)). This will be more involved than the simplest diffusion model, because it involves a couple of different steps. Say we have given training data pairs $(x, y)$, where $x$ is an image and $y$ is a text. We eventually want to generate $x$ given $y$. You need the following components:
1. First you need functions $f(x)$ and $g(y)$ that give you joint embeddings based on image and text. This is achieved via CLIP.
2. Then you need a prior: a mechanism to generate an image embedding from a text embedding. We will do this will a diffusion model.
3. Then you need a decoder: a way to go from an image embedding to an actual image. This is again done with another diffusion model.
4. The actual paper then does up-sampling, which uses further diffusion models, but we won't worry about that.

More details on all these steps are found in my overleaf notes. Also, the approach above is essentially CLIP guidance, but I have not explored classifier-free guidance much, can do that afterwards.

I realized that the above is actually what they do for DALL-E. For GLIDE, especially the CLIP guidance generation, they're not actually training a prior model that generates the image embeddings $z_i$s.