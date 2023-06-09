% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gptneox.R
\name{gptneox}
\alias{gptneox}
\alias{gptneox_from_config}
\alias{gptneox_from_pretrained}
\title{GPT NeoX}
\usage{
gptneox(
  vocab_size = 50432,
  n_embd = 6144,
  n_head = 64,
  n_layer = 44,
  max_pos = 2048,
  n_rot = 0.25
)

gptneox_from_config(identifier, revision = "main")

gptneox_from_pretrained(identifier, revision = "main")
}
\arguments{
\item{vocab_size}{An integer indicating the size of the vocabulary or the number
of unique tokens in the input data.}

\item{n_embd}{An integer specifying the dimensionality of the embedding vectors.}

\item{n_head}{An integer representing the number of attention heads in the
multi-head attention mechanism.}

\item{n_layer}{An integer indicating the number of layers in the deep learning model.}

\item{max_pos}{An integer specifying the maximum position encoding value or
the maximum sequence length.}

\item{n_rot}{An integer indicating the number dimensions used in the rotary
position embedding. Can also be a float \verb{0 < n_rot < 1} indicating the fraction
of \code{n_embd}.}

\item{identifier}{A string representing the identifier or name of the pre-trained
model in the Hugging Face model hub.}

\item{revision}{A string specifying the revision or version of the pre-trained
model in the Hugging Face model hub.}
}
\value{
An initialized \code{\link[torch:nn_module]{torch::nn_module()}}.
}
\description{
Initializes a GPTNeoX like model
}
\section{Functions}{
\itemize{
\item \code{gptneox_from_config()}: Initializes a gptneox model using a configuration defined in HF Hub

\item \code{gptneox_from_pretrained()}: Initializes the gptneox model and load pre-trained weights from HF hub.

}}
