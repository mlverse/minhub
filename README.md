# minhub

<!-- badges: start -->

[![R-CMD-check](https://github.com/mlverse/minhub/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mlverse/minhub/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

minhub is a collection of minimal implementations of deep learning
models inspired by [minGPT]((https://github.com/karpathy/minGPT)). Each
model is designed to be self-contained in a single file with no external
dependencies, making them easy to copy and integrate into your own
projects.

The primary goal of minhub is to provide clean and readable code,
prioritizing simplicity and understanding over speed optimization. These
models are particularly suitable for educational purposes,
experimentation, and as a starting point for more complex
implementations.

Additionally, minhub supports loading weights from pre-trained models
available in the Hugging Face model hub.

## Models

minhub currently implements:

- [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745): [🤗gpt_neox](https://huggingface.co/models?other=gpt_neox), [📄gptneox.R](./R/gptneox.R)
- [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988): [🤗gpt_bigcode](https://huggingface.co/models?other=gpt_bigcode), [📄gptbigcode.R](./R/gptbigcode.R)
- [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/method/gpt-2): [🤗gpt2](https://huggingface.co/models?other=gpt2), [📄gpt2.R](./R/gpt2.R)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971): [🤗llama](https://huggingface.co/models?other=llama), [📄llama.R](./R/llama.R)

## Installation

You can install the development version of minhub like so:

``` r
remotes::install_github("mlverse/minhub")
```

Note:

- dev versions of torch, tok and hfhub are required
- [tok](https://github.com/mlverse/tok) requires a Rust installation. See installation instructions in its repository.

## Examples

