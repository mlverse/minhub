# A GPT2 model, possibly to be used with pre-trained weights from the HuggingFace model hub.
# Documention: https://huggingface.co/docs/transformers/model_doc/gpt2

# Based on the following Python implementations:
# - https://github.com/karpathy/minGPT/blob/master/mingpt/model.py (referred to as "@Karpathy")
# - https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/gpt2/modeling_gpt2.py (referred to as "@Huggingface")

# See also: https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html

# Notes
# - The pre-trained weights do not contain weights for lm_head. That is because these are tied to the input embeddings
#   (see tie_weights() in superclass PretrainedModel(https://github.com/huggingface/transformers/blob/f67dac97bdc63874f2288546b3fa87e69d2ea1c8/src/transformers/modeling_utils.py#L1253),
#   using model-dependent definition of what to tie to here: https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/models/gpt2/modeling_gpt2.py#L735)
#   and @Karpathy actually imports @Huggingface GPTLMHeadModel: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L192
#   Regarding whether to copy or to clone, looking in addition at https://github.com/huggingface/transformers/blob/f67dac97bdc63874f2288546b3fa87e69d2ea1c8/src/transformers/models/gpt2/modeling_gpt2.py#L1007
#   it seems that for gpt2, what effectively happens is `self.lm_head = self.wte`, so they remain tied

#############################         TBD         #############################
# tbd tokenizer: like gpt-neox
# test use generate_sample
# transpose weights???
# doublecheck these prior notes:
  # Daniel do you know why karpathy uses module_dict?
  # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L81
  # also wondering about this line
  # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L88
  # mapped that to mlp forward


#' @noRd
#' @importFrom zeallot %<-%
#' @importFrom purrr map
#' @import torch
NULL

nn_new_gelu <- nn_module(
  forward = function(x) {
    0.5 * x * (1 + torch_tanh(sqrt(2 / pi) * (x + 0.044715 * torch_pow(x, 3))))
  }
)

# Following @Karpathy. See @Huggingface for an alternative implementation.
nn_gpt2_attention <- nn_module(
  initialize = function(n_embd, n_head, n_layer, n_positions, resid_pdrop,
                        attn_pdrop, initializer_range) {
    self$n_head = n_head
    self$n_embd = n_embd
    # key, query, value projections for all heads, but in a batch
    self$c_attn = nn_linear(n_embd, 3 * n_embd)
    # output projection
    self$c_proj = nn_linear(n_embd, n_embd)
    # regularization
    self$attn_dropout = nn_dropout(attn_pdrop)
    self$resid_dropout = nn_dropout(resid_pdrop)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self$register_buffer("bias", torch_tril(torch_ones(n_positions, n_positions))$view(c(1, 1, n_positions, n_positions)))

    nn_init_normal_(self$c_attn$weight, mean = 0, std = initializer_range)
    nn_init_zeros_(self$c_attn$bias)
    nn_init_normal_(self$c_proj$weight, mean = 0, std = initializer_range/sqrt(2 * n_layer))
    nn_init_zeros_(self$c_proj$bias)
  },
  forward = function(x) {
    # batch size, sequence length, embedding dimensionality (n_embd)
    c(B, T, C) %<-% x$shape

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    c(q, k, v)  %<-% self$c_attn(x)$split(self$n_embd, dim = 3) # tbd check dim correct
    k <- k$view(c(B, T, self$n_head, floor(C / self$n_head)))$transpose(2, 3) # (B, nh, T, hs)  # tbd check
    q <- q$view(c(B, T, self$n_head, floor(C / self$n_head)))$transpose(2, 3) # (B, nh, T, hs)
    v <- v$view(c(B, T, self$n_head, floor(C / self$n_head)))$transpose(2, 3) # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att <- q$matmul(k$transpose(-2, -1)) * (1 / sqrt(k$size(-1)))
    att <- att$masked_fill(self$bias[ , , 1:T, 1:T] == 0, Inf)
    att <- F$softmax(att, dim = -1)
    att <- self$attn_dropout(att)
    y <- att$matmul(v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y <- y$transpose(1, 2)$contiguous()$view(c(B, T, C)) # re-assemble all head outputs side by side

    # output projection
    y <- self$resid_dropout(self$c_proj(y))
    y
  }
)

nn_gpt2_mlp <- nn_module(
  initialize = function(n_embd, resid_pdrop, initializer_range) {
    self$c_fc <- nn_linear(n_embd, 4 * n_embd)
    self$c_proj <- nn_linear(4 * n_embd, n_embd)
    self$act <- nn_new_gelu()
    self$dropout <- nn_dropout(resid_pdrop)

    nn_init_normal_(self$c_fc$weight, mean = 0, std = initializer_range)
    nn_init_zeros_(self$c_fc$bias)
    nn_init_normal_(self$c_proj$weight, mean = 0, std = initializer_range)
    nn_init_zeros_(self$c_proj$bias)
  },
  forward = function(x) {
    x |>
      self$c_fc() |>
      self$act() |>
      self$proj() |>
      self$dropout()
  }
)

nn_gpt2_transformer_block <- nn_module(
  initialize = function(n_embd, n_head, n_layer, n_positions, resid_pdrop, attn_pdrop,
                        layer_norm_epsilon, initializer_range) {
    self$ln_1 <- nn_layer_norm(n_embd, layer_norm_epsilon)
    self$attn <- nn_gpt2_attention(n_embd, n_head, n_layer, n_positions, resid_pdrop, attn_pdrop,
                                   initializer_range)
    self$ln_2 <- nn_layer_norm(n_embd, layer_norm_epsilon)
    self$mlp <- nn_gpt2_mlp(n_embd, resid_pdrop, initializer_range)

    nn_init_zeros_(self$ln_1$bias)
    nn_init_ones_(self$ln_1$weight)
    nn_init_zeros_(self$ln_2$bias)
    nn_init_ones_(self$ln_2$weight)
  },
  forward = function(x) {
    x + self$attn(self$ln_1(x)) + self$mlp(self$ln_2(x))
  }
)

nn_gpt2_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_head, n_layer, n_positions, resid_pdrop, embd_pdrop,
                        attn_pdrop, layer_norm_epsilon, initializer_range) {
     self$transformer <- nn_module_dict(list(
     wte = nn_embedding(vocab_size, n_embd),
     wpe = nn_embedding(n_positions, n_embd),
     drop = nn_dropout(embd_pdrop),
     h = nn_sequential(
       !!!1:n_layer |>
         map(\(x) nn_gpt2_transformer_block(n_embd, n_head, n_layer, n_positions, resid_pdrop, attn_pdrop, layer_norm_epsilon, initializer_range))),
     ln_f = nn_layer_norm(n_embd, layer_norm_epsilon)
     ))
    self$lm_head <- nn_linear(n_embd, vocab_size, bias = FALSE)

    # These initializations are in both @Karpathy and @Huggingface (e.g., https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/models/gpt2/modeling_gpt2.py#L455)
    nn_init_normal_(self$transformer$wte$weight, mean = 0, std = initializer_range)
    nn_init_normal_(self$transformer$wte$weight, mean = 0, std = initializer_range)
    nn_init_zeros_(self$transformer$ln_f$bias)
    nn_init_ones_(self$transformer$ln_f$weight)
    nn_init_normal_(self$lm_head$weight, mean = 0, std = initializer_range)
    # The following is both in @Karpathy and in @Huggingface (quote from: https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/models/gpt2/modeling_gpt2.py#LL471C9-L476C111)
    # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
    #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
    #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
    #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
    #
    # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
    for (i in 1:length(self$named_parameters())) {
      pn <- names(self$named_parameters()[i])
      if (grepl("c_proj.weight", pn)) {
        nn_init_normal_(self$named_parameters()[[i]], mean = 0, std = initializer_range/sqrt(2 * n_layer))
      }
    }
  },
  forward = function(x) {
    tok_emb <- self$transformer$wte(x) # token embeddings of shape (b, t, n_embd)
    t <- idx$size()
    pos <- torch_arange(0, t, dtype = torch_long()$unsqueeze(1)) # shape (1, t)
    pos_emb <- self$transformer$wpe(pos) # position embeddings of shape (1, t, n_embd)
    x <- self$transformer$drop(tok_emb + pos_emb)
    x <- self$transformer$h(x)
    x <- self$transformer$ln_f(x)
    x <- self$lm_head(x)
    x
  }
)

#' GPT2
#'
#' Initializes a gpt2-type model
#'
#' @param vocab_size An optional integer indicating the size of the vocabulary or the number of unique tokens in the input data.
#' @param n_embd An integer specifying the Dimensionality of the embeddings and hidden states.
#' @param n_head An integer representing the number of attention heads in each attention layer in the Transformer encoder.
#' @param n_layer An integer indicating the umber of hidden layers in the Transformer encoder.
#' @param n_positions An integer specifying the maximum sequence length that this model might ever be used with.
#' @param resid_pdrop A float specifying dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
#' @param embd_pdrop A float specifying the dropout ratio for the embeddings.
#' @param attn_pdrop A float specifying the dropout ratio for the attention modules.
#' @param layer_norm_epsilon A float specifying t value of epsilon to use in the layer normalization layers.
#' @param initializer_range A float specifying the standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#' @returns An initialized [torch::nn_module()].
#' @export
gpt2 <- function(vocab_size = 50257, n_embd = 768, n_head = 12, n_layer = 12, n_positions = 1024,
                 resid_pdrop = 0.1, embd_pdrop = 0.1, attn_pdrop = 0.1, layer_norm_epsilon = 1e-05,
                 initializer_range = 0.02) {
  nn_gpt2_model(vocab_size, n_embd, n_head, n_layer, n_positions, resid_pdrop, embd_pdrop,
                attn_pdrop, layer_norm_epsilon, initializer_range)
}

gpt2_default_config <- function() {
  config <- list(
    attn_pdrop = 0.1, # dropout ratio for the attention modules
    embd_pdrop = 0.1, # dropout ratio for the embeddings
    initializer_range = 0.02, # standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_epsilon = 1e-05, # the value for epsilon to use in the layer normalization layers
    model_type = gpt2,
    n_embd = 768, # dimensionality of the embeddings and hidden states
    n_head = 12, # number of attention heads in each attention layer in the Transformer encoder
    n_layer = 12, # number of hidden layers in the Transformer encoder
    n_positions = 1024, # maximum sequence length that this model might ever be used with
    resid_pdrop = 0.1, # dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    vocab_size = 50257 #  number of unique tokens in the input data
  )
  config

}

#' @describeIn gpt2 Initializes a gpt2 model using a configuration defined in HF Hub
#' @param identifier A string representing the identifier or name of the pre-trained model in the Hugging Face model hub.
#' @param revision A string specifying the revision or version of the pre-trained model in the Hugging Face model hub.
#' @export
gpt2_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  if (config$model_type != "gpt2")
    cli::cli_abort(c(
      "{.arg config$model_type} must be {.val gpt2}, got {.val {config$model_type}}"
    ))

  default_config <- gpt2_default_config()

  gpt2(vocab_size = if (!is.null(config$vocab_size)) config$vocab_size else default_config$vocab_size,
       n_embd = if (!is.null(config$n_embd)) config$n_embd else default_config$n_embd,
       n_head = if (!is.null(config$n_head)) config$n_head else default_config$n_head,
       n_layer = if (!is.null(config$n_layer)) config$n_layer else default_config$n_layer,
       n_positions = if (!is.null(config$n_positions)) config$n_positions else default_config$n_positions,
       resid_pdrop = if (!is.null(config$resid_pdrop)) config$resid_pdrop else default_config$resid_pdrop,
       embd_pdrop = if (!is.null(config$embd_pdrop)) config$embd_pdrop else default_config$embd_pdrop,
       attn_pdrop = if (!is.null(config$attn_pdrop)) config$attn_pdrop else default_config$attn_pdrop,
       layer_norm_epsilon = if (!is.null(config$layer_norm_epsilon)) config$layer_norm_epsilon else default_config$layer_norm_epsilon,
       initializer_range = if (!is.null(config$initializer_range)) config$initializer_range else default_config$initializer_range
  )
}

#' @describeIn gpt2 Initializes the gpt2 model and load pre-trained weights from HF hub.
#' @param identifier A string representing the identifier or name of the pre-trained model in the Hugging Face model hub.
#' @param revision A string specifying the revision or version of the pre-trained model in the Hugging Face model hub.
#' @export
gpt2_from_pretrained <- function(identifier, revision = "main") {
  with_device(device="meta", {
    model <- gpt2_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- gpt2_hf_weights_remap(state_dict)
  # just an aside here, Daniel what do you think about having load_state_dict() take a parameter "strict" like in PT? (to avoid the error)
  # https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters-from-a-different-model
  state_dict$lm_head.weight <- state_dict$transformer.wte.weight
  # regarding the clone or no question: see notes above, line 15/16
  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

gpt2_hf_weights_remap <- function(state_dict) {
  old_names <- names(state_dict)
  new_names <- paste0("transformer.", old_names)
  names(state_dict) <- new_names
  state_dict
}

