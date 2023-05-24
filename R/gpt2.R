# A GPT2 model, possibly to be used with pre-trained weights from the HuggingFace model hub.
# Documention: https://huggingface.co/docs/transformers/model_doc/gpt2

# Based on the following Python implementations:
# - https://github.com/karpathy/minGPT/blob/master/mingpt/model.py (referred to as "@Karpathy")
# - https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/gpt2/modeling_gpt2.py (referred to as "@Huggingface")

# See also: https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html

# Daniel do you think we need this: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
# what about the generate function? https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L283
# what about these different sizes: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L126

# TBD remap??
# TBD test weight init

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
  initialize = function(n_embd, n_head, n_positions, attn_pdrop, resid_pdrop) {
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
  initialize = function(n_embd, resid_pdrop) {
    self$c_fc <- nn_linear(n_embd, 4 * n_embd)
    self$proj <- nn_linear(4 * n_embd, n_embd)
    self$act <- nn_new_gelu()
    self$dropout <- nn_dropout(resid_pdrop)
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
  initialize = function(n_embd, n_head, n_positions, attn_pdrop, resid_pdrop, layer_norm_epsilon) {
    self$ln_1 <- nn_layer_norm(n_embd, layer_norm_epsilon)
    self$attn <- nn_gpt2_attention(n_embd, n_head, n_positions, attn_pdrop, resid_pdrop)
    self$ln_2 <- nn_layer_norm(n_embd, layer_norm_epsilon)
    # Daniel do you know why karpathy uses module_dict?
    # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L81
    # also wondering about this line
    # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L88
    self$mlp <- nn_gpt2_mlp(n_embd, resid_pdrop)
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
         map(\(x) nn_gpt2_transformer_block(n_embd, n_head, n_positions, attn_pdrop, resid_pdrop, layer_norm_epsilon))),
     ln_f = nn_layer_norm(n_embd, layer_norm_epsilon)
    ))
    self$lm_head <- nn_linear(n_embd, vocab_size, bias = FALSE)
    self$init_weights(self, initializer_range, n_layer)
    # names(module$named_parameters()) == names(module$parameters)
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
  },
  init_weights = function(module, initializer_range, n_layer) {
    module_class <- class(module)[1]
    browser()
    if (module_class == "nn_linear") {
      browser()
      nn_init_normal_(module$weight, mean = 0, std = initializer_range)
      if (!is.null(module$bias)) nn_init_zeros_(module$bias)
    } else if (module_class == "nn_embedding") {
      browser()
      nn_init_normal_(module$weight, mean = 0, std = initializer_range)
    } else if (module_class == "nn_layer_norm") {
      browser()
      nn_init_zeros_(module$bias)
      nn_init_ones_(module$weight)
    }
    for (i in 1:length(module$named_parameters())) {
      pn <- names(module$named_parameters()[i])
      if (grepl("c_proj.weight", pn)) {
        browser()
        nn_init_normal_(p, mean = 0, std = initializer_range/sqrt(2 * n_layer))
      }
    }
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
  browser()
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- purrr::imap(
    gpt2_hf_weights_remap(),
    \(old_name, new_name) state_dict[[old_name]]
  )
  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

gpt2_hf_weights_remap <- function(state_dict) {
  remap <- c(
    transformer.h.11.mlp.d_1.weight = "gpt2.layers.11.mlp.dense_h_to_4h.weight",
    transformer.h.11.mlp.d_1.bias = "gpt2.layers.11.mlp.dense_h_to_4h.bias",
    transformer.h.11.mlp.d_2.weight = "gpt2.layers.11.mlp.dense_4h_to_h.weight",
    transformer.h.11.mlp.d_2.bias = "gpt2.layers.11.mlp.dense_4h_to_h.bias",
    transformer.h.12.ln_1.weight = "gpt2.layers.12.input_layernorm.weight",
    transformer.h.12.ln_1.bias = "gpt2.layers.12.input_layernorm.bias",
    transformer.h.12.ln_2.weight = "gpt2.layers.12.post_attention_layernorm.weight",
    transformer.h.12.ln_2.bias = "gpt2.layers.12.post_attention_layernorm.bias",
    transformer.h.12.attn.bias = "gpt2.layers.12.attention.bias",
    transformer.h.12.attn.masked_bias = "gpt2.layers.12.attention.masked_bias",
    transformer.h.12.attn.rotary.inv_freq = "gpt2.layers.12.attention.rotary_emb.inv_freq",
    transformer.h.12.attn.c_attn.weight = "gpt2.layers.12.attention.query_key_value.weight",
    transformer.h.12.attn.c_attn.bias = "gpt2.layers.12.attention.query_key_value.bias",
    transformer.h.12.attn.c_proj.weight = "gpt2.layers.12.attention.dense.weight",
    transformer.h.12.attn.c_proj.bias = "gpt2.layers.12.attention.dense.bias",
    transformer.h.12.mlp.d_1.weight = "gpt2.layers.12.mlp.dense_h_to_4h.weight",
    transformer.h.12.mlp.d_1.bias = "gpt2.layers.12.mlp.dense_h_to_4h.bias",
    transformer.h.12.mlp.d_2.weight = "gpt2.layers.12.mlp.dense_4h_to_h.weight",
    transformer.h.12.mlp.d_2.bias = "gpt2.layers.12.mlp.dense_4h_to_h.bias",
    transformer.h.13.ln_1.weight = "gpt2.layers.13.input_layernorm.weight",
    transformer.h.13.ln_1.bias = "gpt2.layers.13.input_layernorm.bias",
    transformer.h.13.ln_2.weight = "gpt2.layers.13.post_attention_layernorm.weight",
    transformer.h.13.ln_2.bias = "gpt2.layers.13.post_attention_layernorm.bias",
    transformer.h.13.attn.bias = "gpt2.layers.13.attention.bias",
    transformer.h.13.attn.masked_bias = "gpt2.layers.13.attention.masked_bias",
    transformer.h.13.attn.rotary.inv_freq = "gpt2.layers.13.attention.rotary_emb.inv_freq",
    transformer.h.13.attn.c_attn.weight = "gpt2.layers.13.attention.query_key_value.weight",
    transformer.h.13.attn.c_attn.bias = "gpt2.layers.13.attention.query_key_value.bias",
    transformer.h.13.attn.c_proj.weight = "gpt2.layers.13.attention.dense.weight",
    transformer.h.13.attn.c_proj.bias = "gpt2.layers.13.attention.dense.bias",
    transformer.h.13.mlp.d_1.weight = "gpt2.layers.13.mlp.dense_h_to_4h.weight",
    transformer.h.13.mlp.d_1.bias = "gpt2.layers.13.mlp.dense_h_to_4h.bias",
    transformer.h.13.mlp.d_2.weight = "gpt2.layers.13.mlp.dense_4h_to_h.weight",
    transformer.h.13.mlp.d_2.bias = "gpt2.layers.13.mlp.dense_4h_to_h.bias",
    transformer.h.14.ln_1.weight = "gpt2.layers.14.input_layernorm.weight",
    transformer.h.14.ln_1.bias = "gpt2.layers.14.input_layernorm.bias",
    transformer.h.14.ln_2.weight = "gpt2.layers.14.post_attention_layernorm.weight",
    transformer.h.14.ln_2.bias = "gpt2.layers.14.post_attention_layernorm.bias",
    transformer.h.14.attn.bias = "gpt2.layers.14.attention.bias",
    transformer.h.14.attn.masked_bias = "gpt2.layers.14.attention.masked_bias",
    transformer.h.14.attn.rotary.inv_freq = "gpt2.layers.14.attention.rotary_emb.inv_freq",
    transformer.h.14.attn.c_attn.weight = "gpt2.layers.14.attention.query_key_value.weight",
    transformer.h.14.attn.c_attn.bias = "gpt2.layers.14.attention.query_key_value.bias",
    transformer.h.14.attn.c_proj.weight = "gpt2.layers.14.attention.dense.weight",
    transformer.h.14.attn.c_proj.bias = "gpt2.layers.14.attention.dense.bias",
    transformer.h.14.mlp.d_1.weight = "gpt2.layers.14.mlp.dense_h_to_4h.weight",
    transformer.h.14.mlp.d_1.bias = "gpt2.layers.14.mlp.dense_h_to_4h.bias",
    transformer.h.14.mlp.d_2.weight = "gpt2.layers.14.mlp.dense_4h_to_h.weight",
    transformer.h.14.mlp.d_2.bias = "gpt2.layers.14.mlp.dense_4h_to_h.bias",
    transformer.h.15.ln_1.weight = "gpt2.layers.15.input_layernorm.weight",
    transformer.h.15.ln_1.bias = "gpt2.layers.15.input_layernorm.bias",
    transformer.h.15.ln_2.weight = "gpt2.layers.15.post_attention_layernorm.weight",
    transformer.h.15.ln_2.bias = "gpt2.layers.15.post_attention_layernorm.bias",
    transformer.h.15.attn.bias = "gpt2.layers.15.attention.bias",
    transformer.h.15.attn.masked_bias = "gpt2.layers.15.attention.masked_bias",
    transformer.h.15.attn.rotary.inv_freq = "gpt2.layers.15.attention.rotary_emb.inv_freq",
    transformer.h.15.attn.c_attn.weight = "gpt2.layers.15.attention.query_key_value.weight",
    transformer.h.15.attn.c_attn.bias = "gpt2.layers.15.attention.query_key_value.bias",
    transformer.h.15.attn.c_proj.weight = "gpt2.layers.15.attention.dense.weight",
    transformer.h.15.attn.c_proj.bias = "gpt2.layers.15.attention.dense.bias",
    transformer.h.15.mlp.d_1.weight = "gpt2.layers.15.mlp.dense_h_to_4h.weight",
    transformer.h.15.mlp.d_1.bias = "gpt2.layers.15.mlp.dense_h_to_4h.bias",
    transformer.h.15.mlp.d_2.weight = "gpt2.layers.15.mlp.dense_4h_to_h.weight",
    transformer.h.15.mlp.d_2.bias = "gpt2.layers.15.mlp.dense_4h_to_h.bias",
    transformer.ln_f.weight = "gpt2.final_layer_norm.weight",
    transformer.ln_f.bias = "gpt2.final_layer_norm.bias",
    lm_head.weight = "embed_out.weight",
    transformer.wte.weight = "gpt2.embed_in.weight",
    transformer.h.0.ln_1.weight = "gpt2.layers.0.input_layernorm.weight",
    transformer.h.0.ln_1.bias = "gpt2.layers.0.input_layernorm.bias",
    transformer.h.0.ln_2.weight = "gpt2.layers.0.post_attention_layernorm.weight",
    transformer.h.0.ln_2.bias = "gpt2.layers.0.post_attention_layernorm.bias",
    transformer.h.0.attn.bias = "gpt2.layers.0.attention.bias",
    transformer.h.0.attn.masked_bias = "gpt2.layers.0.attention.masked_bias",
    transformer.h.0.attn.rotary.inv_freq = "gpt2.layers.0.attention.rotary_emb.inv_freq",
    transformer.h.0.attn.c_attn.weight = "gpt2.layers.0.attention.query_key_value.weight",
    transformer.h.0.attn.c_attn.bias = "gpt2.layers.0.attention.query_key_value.bias",
    transformer.h.0.attn.c_proj.weight = "gpt2.layers.0.attention.dense.weight",
    transformer.h.0.attn.c_proj.bias = "gpt2.layers.0.attention.dense.bias",
    transformer.h.0.mlp.d_1.weight = "gpt2.layers.0.mlp.dense_h_to_4h.weight",
    transformer.h.0.mlp.d_1.bias = "gpt2.layers.0.mlp.dense_h_to_4h.bias",
    transformer.h.0.mlp.d_2.weight = "gpt2.layers.0.mlp.dense_4h_to_h.weight",
    transformer.h.0.mlp.d_2.bias = "gpt2.layers.0.mlp.dense_4h_to_h.bias",
    transformer.h.1.ln_1.weight = "gpt2.layers.1.input_layernorm.weight",
    transformer.h.1.ln_1.bias = "gpt2.layers.1.input_layernorm.bias",
    transformer.h.1.ln_2.weight = "gpt2.layers.1.post_attention_layernorm.weight",
    transformer.h.1.ln_2.bias = "gpt2.layers.1.post_attention_layernorm.bias",
    transformer.h.1.attn.bias = "gpt2.layers.1.attention.bias",
    transformer.h.1.attn.masked_bias = "gpt2.layers.1.attention.masked_bias",
    transformer.h.1.attn.rotary.inv_freq = "gpt2.layers.1.attention.rotary_emb.inv_freq",
    transformer.h.1.attn.c_attn.weight = "gpt2.layers.1.attention.query_key_value.weight",
    transformer.h.1.attn.c_attn.bias = "gpt2.layers.1.attention.query_key_value.bias",
    transformer.h.1.attn.c_proj.weight = "gpt2.layers.1.attention.dense.weight",
    transformer.h.1.attn.c_proj.bias = "gpt2.layers.1.attention.dense.bias",
    transformer.h.1.mlp.d_1.weight = "gpt2.layers.1.mlp.dense_h_to_4h.weight",
    transformer.h.1.mlp.d_1.bias = "gpt2.layers.1.mlp.dense_h_to_4h.bias",
    transformer.h.1.mlp.d_2.weight = "gpt2.layers.1.mlp.dense_4h_to_h.weight",
    transformer.h.1.mlp.d_2.bias = "gpt2.layers.1.mlp.dense_4h_to_h.bias",
    transformer.h.2.ln_1.weight = "gpt2.layers.2.input_layernorm.weight",
    transformer.h.2.ln_1.bias = "gpt2.layers.2.input_layernorm.bias",
    transformer.h.2.ln_2.weight = "gpt2.layers.2.post_attention_layernorm.weight",
    transformer.h.2.ln_2.bias = "gpt2.layers.2.post_attention_layernorm.bias",
    transformer.h.2.attn.bias = "gpt2.layers.2.attention.bias",
    transformer.h.2.attn.masked_bias = "gpt2.layers.2.attention.masked_bias",
    transformer.h.2.attn.rotary.inv_freq = "gpt2.layers.2.attention.rotary_emb.inv_freq",
    transformer.h.2.attn.c_attn.weight = "gpt2.layers.2.attention.query_key_value.weight",
    transformer.h.2.attn.c_attn.bias = "gpt2.layers.2.attention.query_key_value.bias",
    transformer.h.2.attn.c_proj.weight = "gpt2.layers.2.attention.dense.weight",
    transformer.h.2.attn.c_proj.bias = "gpt2.layers.2.attention.dense.bias",
    transformer.h.2.mlp.d_1.weight = "gpt2.layers.2.mlp.dense_h_to_4h.weight",
    transformer.h.2.mlp.d_1.bias = "gpt2.layers.2.mlp.dense_h_to_4h.bias",
    transformer.h.2.mlp.d_2.weight = "gpt2.layers.2.mlp.dense_4h_to_h.weight",
    transformer.h.2.mlp.d_2.bias = "gpt2.layers.2.mlp.dense_4h_to_h.bias",
    transformer.h.3.ln_1.weight = "gpt2.layers.3.input_layernorm.weight",
    transformer.h.3.ln_1.bias = "gpt2.layers.3.input_layernorm.bias",
    transformer.h.3.ln_2.weight = "gpt2.layers.3.post_attention_layernorm.weight",
    transformer.h.3.ln_2.bias = "gpt2.layers.3.post_attention_layernorm.bias",
    transformer.h.3.attn.bias = "gpt2.layers.3.attention.bias",
    transformer.h.3.attn.masked_bias = "gpt2.layers.3.attention.masked_bias",
    transformer.h.3.attn.rotary.inv_freq = "gpt2.layers.3.attention.rotary_emb.inv_freq",
    transformer.h.3.attn.c_attn.weight = "gpt2.layers.3.attention.query_key_value.weight",
    transformer.h.3.attn.c_attn.bias = "gpt2.layers.3.attention.query_key_value.bias",
    transformer.h.3.attn.c_proj.weight = "gpt2.layers.3.attention.dense.weight",
    transformer.h.3.attn.c_proj.bias = "gpt2.layers.3.attention.dense.bias",
    transformer.h.3.mlp.d_1.weight = "gpt2.layers.3.mlp.dense_h_to_4h.weight",
    transformer.h.3.mlp.d_1.bias = "gpt2.layers.3.mlp.dense_h_to_4h.bias",
    transformer.h.3.mlp.d_2.weight = "gpt2.layers.3.mlp.dense_4h_to_h.weight",
    transformer.h.3.mlp.d_2.bias = "gpt2.layers.3.mlp.dense_4h_to_h.bias",
    transformer.h.4.ln_1.weight = "gpt2.layers.4.input_layernorm.weight",
    transformer.h.4.ln_1.bias = "gpt2.layers.4.input_layernorm.bias",
    transformer.h.4.ln_2.weight = "gpt2.layers.4.post_attention_layernorm.weight",
    transformer.h.4.ln_2.bias = "gpt2.layers.4.post_attention_layernorm.bias",
    transformer.h.4.attn.bias = "gpt2.layers.4.attention.bias",
    transformer.h.4.attn.masked_bias = "gpt2.layers.4.attention.masked_bias",
    transformer.h.4.attn.rotary.inv_freq = "gpt2.layers.4.attention.rotary_emb.inv_freq",
    transformer.h.4.attn.c_attn.weight = "gpt2.layers.4.attention.query_key_value.weight",
    transformer.h.4.attn.c_attn.bias = "gpt2.layers.4.attention.query_key_value.bias",
    transformer.h.4.attn.c_proj.weight = "gpt2.layers.4.attention.dense.weight",
    transformer.h.4.attn.c_proj.bias = "gpt2.layers.4.attention.dense.bias",
    transformer.h.4.mlp.d_1.weight = "gpt2.layers.4.mlp.dense_h_to_4h.weight",
    transformer.h.4.mlp.d_1.bias = "gpt2.layers.4.mlp.dense_h_to_4h.bias",
    transformer.h.4.mlp.d_2.weight = "gpt2.layers.4.mlp.dense_4h_to_h.weight",
    transformer.h.4.mlp.d_2.bias = "gpt2.layers.4.mlp.dense_4h_to_h.bias",
    transformer.h.5.ln_1.weight = "gpt2.layers.5.input_layernorm.weight",
    transformer.h.5.ln_1.bias = "gpt2.layers.5.input_layernorm.bias",
    transformer.h.5.ln_2.weight = "gpt2.layers.5.post_attention_layernorm.weight",
    transformer.h.5.ln_2.bias = "gpt2.layers.5.post_attention_layernorm.bias",
    transformer.h.5.attn.bias = "gpt2.layers.5.attention.bias",
    transformer.h.5.attn.masked_bias = "gpt2.layers.5.attention.masked_bias",
    transformer.h.5.attn.rotary.inv_freq = "gpt2.layers.5.attention.rotary_emb.inv_freq",
    transformer.h.5.attn.c_attn.weight = "gpt2.layers.5.attention.query_key_value.weight",
    transformer.h.5.attn.c_attn.bias = "gpt2.layers.5.attention.query_key_value.bias",
    transformer.h.5.attn.c_proj.weight = "gpt2.layers.5.attention.dense.weight",
    transformer.h.5.attn.c_proj.bias = "gpt2.layers.5.attention.dense.bias",
    transformer.h.5.mlp.d_1.weight = "gpt2.layers.5.mlp.dense_h_to_4h.weight",
    transformer.h.5.mlp.d_1.bias = "gpt2.layers.5.mlp.dense_h_to_4h.bias",
    transformer.h.5.mlp.d_2.weight = "gpt2.layers.5.mlp.dense_4h_to_h.weight",
    transformer.h.5.mlp.d_2.bias = "gpt2.layers.5.mlp.dense_4h_to_h.bias",
    transformer.h.6.ln_1.weight = "gpt2.layers.6.input_layernorm.weight",
    transformer.h.6.ln_1.bias = "gpt2.layers.6.input_layernorm.bias",
    transformer.h.6.ln_2.weight = "gpt2.layers.6.post_attention_layernorm.weight",
    transformer.h.6.ln_2.bias = "gpt2.layers.6.post_attention_layernorm.bias",
    transformer.h.6.attn.bias = "gpt2.layers.6.attention.bias",
    transformer.h.6.attn.masked_bias = "gpt2.layers.6.attention.masked_bias",
    transformer.h.6.attn.rotary.inv_freq = "gpt2.layers.6.attention.rotary_emb.inv_freq",
    transformer.h.6.attn.c_attn.weight = "gpt2.layers.6.attention.query_key_value.weight",
    transformer.h.6.attn.c_attn.bias = "gpt2.layers.6.attention.query_key_value.bias",
    transformer.h.6.attn.c_proj.weight = "gpt2.layers.6.attention.dense.weight",
    transformer.h.6.attn.c_proj.bias = "gpt2.layers.6.attention.dense.bias",
    transformer.h.6.mlp.d_1.weight = "gpt2.layers.6.mlp.dense_h_to_4h.weight",
    transformer.h.6.mlp.d_1.bias = "gpt2.layers.6.mlp.dense_h_to_4h.bias",
    transformer.h.6.mlp.d_2.weight = "gpt2.layers.6.mlp.dense_4h_to_h.weight",
    transformer.h.6.mlp.d_2.bias = "gpt2.layers.6.mlp.dense_4h_to_h.bias",
    transformer.h.7.ln_1.weight = "gpt2.layers.7.input_layernorm.weight",
    transformer.h.7.ln_1.bias = "gpt2.layers.7.input_layernorm.bias",
    transformer.h.7.ln_2.weight = "gpt2.layers.7.post_attention_layernorm.weight",
    transformer.h.7.ln_2.bias = "gpt2.layers.7.post_attention_layernorm.bias",
    transformer.h.7.attn.bias = "gpt2.layers.7.attention.bias",
    transformer.h.7.attn.masked_bias = "gpt2.layers.7.attention.masked_bias",
    transformer.h.7.attn.rotary.inv_freq = "gpt2.layers.7.attention.rotary_emb.inv_freq",
    transformer.h.7.attn.c_attn.weight = "gpt2.layers.7.attention.query_key_value.weight",
    transformer.h.7.attn.c_attn.bias = "gpt2.layers.7.attention.query_key_value.bias",
    transformer.h.7.attn.c_proj.weight = "gpt2.layers.7.attention.dense.weight",
    transformer.h.7.attn.c_proj.bias = "gpt2.layers.7.attention.dense.bias",
    transformer.h.7.mlp.d_1.weight = "gpt2.layers.7.mlp.dense_h_to_4h.weight",
    transformer.h.7.mlp.d_1.bias = "gpt2.layers.7.mlp.dense_h_to_4h.bias",
    transformer.h.7.mlp.d_2.weight = "gpt2.layers.7.mlp.dense_4h_to_h.weight",
    transformer.h.7.mlp.d_2.bias = "gpt2.layers.7.mlp.dense_4h_to_h.bias",
    transformer.h.8.ln_1.weight = "gpt2.layers.8.input_layernorm.weight",
    transformer.h.8.ln_1.bias = "gpt2.layers.8.input_layernorm.bias",
    transformer.h.8.ln_2.weight = "gpt2.layers.8.post_attention_layernorm.weight",
    transformer.h.8.ln_2.bias = "gpt2.layers.8.post_attention_layernorm.bias",
    transformer.h.8.attn.bias = "gpt2.layers.8.attention.bias",
    transformer.h.8.attn.masked_bias = "gpt2.layers.8.attention.masked_bias",
    transformer.h.8.attn.rotary.inv_freq = "gpt2.layers.8.attention.rotary_emb.inv_freq",
    transformer.h.8.attn.c_attn.weight = "gpt2.layers.8.attention.query_key_value.weight",
    transformer.h.8.attn.c_attn.bias = "gpt2.layers.8.attention.query_key_value.bias",
    transformer.h.8.attn.c_proj.weight = "gpt2.layers.8.attention.dense.weight",
    transformer.h.8.attn.c_proj.bias = "gpt2.layers.8.attention.dense.bias",
    transformer.h.8.mlp.d_1.weight = "gpt2.layers.8.mlp.dense_h_to_4h.weight",
    transformer.h.8.mlp.d_1.bias = "gpt2.layers.8.mlp.dense_h_to_4h.bias",
    transformer.h.8.mlp.d_2.weight = "gpt2.layers.8.mlp.dense_4h_to_h.weight",
    transformer.h.8.mlp.d_2.bias = "gpt2.layers.8.mlp.dense_4h_to_h.bias",
    transformer.h.9.ln_1.weight = "gpt2.layers.9.input_layernorm.weight",
    transformer.h.9.ln_1.bias = "gpt2.layers.9.input_layernorm.bias",
    transformer.h.9.ln_2.weight = "gpt2.layers.9.post_attention_layernorm.weight",
    transformer.h.9.ln_2.bias = "gpt2.layers.9.post_attention_layernorm.bias",
    transformer.h.9.attn.bias = "gpt2.layers.9.attention.bias",
    transformer.h.9.attn.masked_bias = "gpt2.layers.9.attention.masked_bias",
    transformer.h.9.attn.rotary.inv_freq = "gpt2.layers.9.attention.rotary_emb.inv_freq",
    transformer.h.9.attn.c_attn.weight = "gpt2.layers.9.attention.query_key_value.weight",
    transformer.h.9.attn.c_attn.bias = "gpt2.layers.9.attention.query_key_value.bias",
    transformer.h.9.attn.c_proj.weight = "gpt2.layers.9.attention.dense.weight",
    transformer.h.9.attn.c_proj.bias = "gpt2.layers.9.attention.dense.bias",
    transformer.h.9.mlp.d_1.weight = "gpt2.layers.9.mlp.dense_h_to_4h.weight",
    transformer.h.9.mlp.d_1.bias = "gpt2.layers.9.mlp.dense_h_to_4h.bias",
    transformer.h.9.mlp.d_2.weight = "gpt2.layers.9.mlp.dense_4h_to_h.weight",
    transformer.h.9.mlp.d_2.bias = "gpt2.layers.9.mlp.dense_4h_to_h.bias",
    transformer.h.10.ln_1.weight = "gpt2.layers.10.input_layernorm.weight",
    transformer.h.10.ln_1.bias = "gpt2.layers.10.input_layernorm.bias",
    transformer.h.10.ln_2.weight = "gpt2.layers.10.post_attention_layernorm.weight",
    transformer.h.10.ln_2.bias = "gpt2.layers.10.post_attention_layernorm.bias",
    transformer.h.10.attn.bias = "gpt2.layers.10.attention.bias",
    transformer.h.10.attn.masked_bias = "gpt2.layers.10.attention.masked_bias",
    transformer.h.10.attn.rotary.inv_freq = "gpt2.layers.10.attention.rotary_emb.inv_freq",
    transformer.h.10.attn.c_attn.weight = "gpt2.layers.10.attention.query_key_value.weight",
    transformer.h.10.attn.c_attn.bias = "gpt2.layers.10.attention.query_key_value.bias",
    transformer.h.10.attn.c_proj.weight = "gpt2.layers.10.attention.dense.weight",
    transformer.h.10.attn.c_proj.bias = "gpt2.layers.10.attention.dense.bias",
    transformer.h.10.mlp.d_1.weight = "gpt2.layers.10.mlp.dense_h_to_4h.weight",
    transformer.h.10.mlp.d_1.bias = "gpt2.layers.10.mlp.dense_h_to_4h.bias",
    transformer.h.10.mlp.d_2.weight = "gpt2.layers.10.mlp.dense_4h_to_h.weight",
    transformer.h.10.mlp.d_2.bias = "gpt2.layers.10.mlp.dense_4h_to_h.bias",
    transformer.h.11.ln_1.weight = "gpt2.layers.11.input_layernorm.weight",
    transformer.h.11.ln_1.bias = "gpt2.layers.11.input_layernorm.bias",
    transformer.h.11.ln_2.weight = "gpt2.layers.11.post_attention_layernorm.weight",
    transformer.h.11.ln_2.bias = "gpt2.layers.11.post_attention_layernorm.bias",
    transformer.h.11.attn.bias = "gpt2.layers.11.attention.bias",
    transformer.h.11.attn.masked_bias = "gpt2.layers.11.attention.masked_bias",
    transformer.h.11.attn.rotary.inv_freq = "gpt2.layers.11.attention.rotary_emb.inv_freq",
    transformer.h.11.attn.c_attn.weight = "gpt2.layers.11.attention.query_key_value.weight",
    transformer.h.11.attn.c_attn.bias = "gpt2.layers.11.attention.query_key_value.bias",
    transformer.h.11.attn.c_proj.weight = "gpt2.layers.11.attention.dense.weight",
    transformer.h.11.attn.c_proj.bias = "gpt2.layers.11.attention.dense.bias"
  )
}

