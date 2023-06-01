# References:
# - https://github.com/karpathy/minGPT
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py

#' @noRd
#' @importFrom zeallot %<-%
#' @importFrom purrr map
#' @import torch
NULL

rotate_half <- function(x) {
  c(x1, x2) %<-% torch_split(x, x$size(-1) / 2, -1)
  torch_cat(list(-x2, x1), dim = -1)
}

nn_rotary_embedding <- nn_module(
  initialize = function(n_rot, max_pos, base=10000) {
    self$n_rot <- n_rot
    self$max_pos <- max_pos

    self$inv_freq <- nn_buffer(
      torch_ones(1) / (base ^ (torch_arange(0, n_rot-1, step = 2) / n_rot)),
      persistent = TRUE
    )

    self$cached_embeddings() # populate the cache
  },
  .load_from_state_dict = function(...) {
    #loading a new state dict invalidates the cache
    super$.load_from_state_dict(...)
    self$cached_embeddings(invalidate = TRUE)
  },
  cached_embeddings = function(t = 1, invalidate = FALSE) {
    invalidate <- invalidate || is.null(self$cos)
    if (invalidate) {
      freqs <- torch_arange(start = 0, end = self$max_pos - 1)$
        float()$
        outer(self$inv_freq)$
        view(c(1,1, self$max_pos, self$n_rot/2))
      emb <- torch_cat(list(freqs, freqs), dim = -1)
      self$cos <- nn_buffer(emb$cos(), persistent = FALSE)
      self$sin <- nn_buffer(emb$sin(), persistent = FALSE)
    }
    list(self$cos[,,1:t,], self$sin[,,1:t,])
  },
  forward = function(x) {
    c(b, nh, t, ed) %<-% x$shape
    c(cos, sin) %<-% self$cached_embeddings(t)

    # rotary embeddings are applied only to the first `n_rot` dims of x
    c(x_rot, x) %<-% x$split(c(self$n_rot, ed - self$n_rot), dim = -1)
    x_rot <- x_rot * cos + rotate_half(x_rot) * sin

    torch_cat(list(x_rot, x), dim = -1)
  }
)

nn_gptneox_attention <- nn_module(
  initialize = function(n_head, n_embd, max_pos, n_rot) {
    self$n_head <- n_head
    self$n_embd <- n_embd

    self$max_pos <- max_pos
    self$n_rot <- if (n_rot <= 1) n_rot * (n_embd / n_head) else n_rot

    self$c_attn <- nn_linear(n_embd, 3*n_embd)
    self$c_proj <- nn_linear(n_embd, n_embd)
    self$rotary <- nn_rotary_embedding(self$n_rot, max_pos)

    # causal mask to ensure that attention is only applied to the left in the
    # input sequence
    self$bias <- torch_ones(max_pos, max_pos)$
      bool()$
      tril()$
      view(c(1, 1, max_pos, max_pos)) |>
      nn_buffer()

    self$masked_bias <- nn_buffer(torch_tensor(-Inf))
  },
  forward = function(x) {
    c(b, t, h) %<-% x$shape
    # (b, t, h) -> [(b, nh, t, h/nh) * 3]
    c(q, k, v) %<-% (self$c_attn(x)$
      view(c(b, t, self$n_head, self$n_embd / self$n_head * 3))$
      split(self$n_embd / self$n_head, dim = -1) |>
      map(\(x) x$transpose(2, 3)))

    q <- self$rotary(q)$to(dtype="float")
    k <- self$rotary(k)$to(dtype="float")

    # the following block requires key and value to be in float32 otherwise
    # it leads to precision problems
    att <- torch_matmul(q, k$transpose(-2, -1)) * (1 / sqrt(k$size(-1)))
    att <- att$masked_fill(self$bias[,,1:t, 1:t] == 0, self$masked_bias)
    att <- nnf_softmax(att, dim=-1)$to(dtype = v$dtype)
    
    y <- torch_matmul(att, v)$transpose(2, 3)$contiguous()$view(c(b, t, h))
    self$c_proj(y)
  }
)

nn_gptneox_mlp <- nn_module(
  initialize = function(n_embd, n_inter = 4*n_embd) {
    self$d_1 <- nn_linear(n_embd, n_inter)
    self$d_2 <- nn_linear(n_inter, n_embd)
    self$act <- nn_gelu()
  },
  forward = function(x) {
    x |>
      self$d_1() |>
      self$act() |>
      self$d_2()
  }
)

nn_gptneox_layer <- nn_module(
  initialize = function(n_embd, n_head, max_pos, n_rot) {
    self$ln_1 <- nn_layer_norm(n_embd)
    self$ln_2 <- nn_layer_norm(n_embd)
    self$attn <- nn_gptneox_attention(n_head, n_embd, max_pos, n_rot)
    self$mlp <- nn_gptneox_mlp(n_embd)
  },
  forward = function(x) {
    x + self$attn(self$ln_1(x)) + self$mlp(self$ln_2(x))
  }
)

nn_gptneox_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_head, n_layer, max_pos, n_rot) {
    self$transformer <- nn_module_dict(list(
     wte = nn_embedding(vocab_size, n_embd),
     h = nn_sequential(!!!map(
       1:n_layer,
       \(x) nn_gptneox_layer(n_embd, n_head, max_pos, n_rot)
     )),
     ln_f = nn_layer_norm(n_embd)
    ))
    self$lm_head <- nn_linear(n_embd, vocab_size, bias = FALSE)
  },
  forward = function(idx) {
    x <- self$transformer$wte(idx)
    x <- self$transformer$h(x)
    x <- self$transformer$ln_f(x)
    self$lm_head(x)
  }
)

#' GPT NeoX
#'
#' Initializes a GPTNeoX like model
#'
#' @param vocab_size An integer indicating the size of the vocabulary or the number
#'   of unique tokens in the input data.
#' @param n_embd An integer specifying the dimensionality of the embedding vectors.
#' @param n_head An integer representing the number of attention heads in the
#'   multi-head attention mechanism.
#' @param n_layer An integer indicating the number of layers in the deep learning model.
#' @param max_pos An integer specifying the maximum position encoding value or
#'  the maximum sequence length.
#' @param n_rot An integer indicating the number dimensions used in the rotary
#'  position embedding. Can also be a float `0 < n_rot < 1` indicating the fraction
#'  of `n_embd`.
#' @param identifier A string representing the identifier or name of the pre-trained
#'  model in the Hugging Face model hub.
#' @param revision A string specifying the revision or version of the pre-trained
#'  model in the Hugging Face model hub.
#' @returns An initialized [torch::nn_module()].
#' @export
gptneox <- function(vocab_size=50432, n_embd=6144, n_head=64, n_layer=44,
                    max_pos=2048, n_rot=0.25) {
  nn_gptneox_model(vocab_size, n_embd, n_head, n_layer, max_pos, n_rot)
}

#' @describeIn gptneox Initializes a gptneox model using a configuration defined in HF Hub
#' @export
gptneox_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  if (config$model_type != "gpt_neox")
    cli::cli_abort(c(
      "{.arg config$model_type} must be {.val gpt_neox}, got {.val {config$model_type}}"
    ))

  # parallel residual is not supported
  if (!config$use_parallel_residual)
    cli::cli_abort(c(
      x = "Non parallel residual is not supported.",
      i = "{.arg config$use_parallel_residual} is {.val FALSE}"
    ))

  if (config$hidden_act != "gelu")
    cli::cli_abort(c(
      x = "Unsupported {.arg config$hidden_act}: {.val {config$hidden_act}}",
      i = "Currently only {.val gelu} is supported."
    ))

  if ((config$intermediate_size / config$hidden_size) != 4)
    cli::cli_abort(c(
      x = "{.arg config$intermediate_size} must be 4*{.arg config$hidden_size}",
      i = "Got {.val {config$intermediate_size}} and {.val {config$hidden_size}}"
    ))

  if (config$layer_norm_eps != 1e-5)
    cli::cli_abort(c(
      x = "{.arg config$layer_norm_eps} must be 1e-5, got {.val {config$layer_norm_eps}}"
    ))

  # remap HF config attributes to minhub configurations
  vocab_size <- config$vocab_size
  n_embd     <- config$hidden_size
  n_head     <- config$num_attention_heads
  n_layer    <- config$num_hidden_layers
  max_pos    <- config$max_position_embeddings
  n_rot      <- config$rotary_pct

  gptneox(vocab_size, n_embd, n_head, n_layer, max_pos, n_rot)
}

#' @describeIn gptneox Initializes the gptneox model and load pre-trained weights from HF hub.
#' @export
gptneox_from_pretrained <- function(identifier, revision = "main") {
  with_device(device="meta", {
    model <- gptneox_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- purrr::imap(
    gptneox_hf_weights_remap(),
    \(old_name, new_name) state_dict[[old_name]]
  )
  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

gptneox_hf_weights_remap <- function(state_dict) {
  remap <- c(
    transformer.h.11.mlp.d_1.weight = "gpt_neox.layers.11.mlp.dense_h_to_4h.weight",
    transformer.h.11.mlp.d_1.bias = "gpt_neox.layers.11.mlp.dense_h_to_4h.bias",
    transformer.h.11.mlp.d_2.weight = "gpt_neox.layers.11.mlp.dense_4h_to_h.weight",
    transformer.h.11.mlp.d_2.bias = "gpt_neox.layers.11.mlp.dense_4h_to_h.bias",
    transformer.h.12.ln_1.weight = "gpt_neox.layers.12.input_layernorm.weight",
    transformer.h.12.ln_1.bias = "gpt_neox.layers.12.input_layernorm.bias",
    transformer.h.12.ln_2.weight = "gpt_neox.layers.12.post_attention_layernorm.weight",
    transformer.h.12.ln_2.bias = "gpt_neox.layers.12.post_attention_layernorm.bias",
    transformer.h.12.attn.bias = "gpt_neox.layers.12.attention.bias",
    transformer.h.12.attn.masked_bias = "gpt_neox.layers.12.attention.masked_bias",
    transformer.h.12.attn.rotary.inv_freq = "gpt_neox.layers.12.attention.rotary_emb.inv_freq",
    transformer.h.12.attn.c_attn.weight = "gpt_neox.layers.12.attention.query_key_value.weight",
    transformer.h.12.attn.c_attn.bias = "gpt_neox.layers.12.attention.query_key_value.bias",
    transformer.h.12.attn.c_proj.weight = "gpt_neox.layers.12.attention.dense.weight",
    transformer.h.12.attn.c_proj.bias = "gpt_neox.layers.12.attention.dense.bias",
    transformer.h.12.mlp.d_1.weight = "gpt_neox.layers.12.mlp.dense_h_to_4h.weight",
    transformer.h.12.mlp.d_1.bias = "gpt_neox.layers.12.mlp.dense_h_to_4h.bias",
    transformer.h.12.mlp.d_2.weight = "gpt_neox.layers.12.mlp.dense_4h_to_h.weight",
    transformer.h.12.mlp.d_2.bias = "gpt_neox.layers.12.mlp.dense_4h_to_h.bias",
    transformer.h.13.ln_1.weight = "gpt_neox.layers.13.input_layernorm.weight",
    transformer.h.13.ln_1.bias = "gpt_neox.layers.13.input_layernorm.bias",
    transformer.h.13.ln_2.weight = "gpt_neox.layers.13.post_attention_layernorm.weight",
    transformer.h.13.ln_2.bias = "gpt_neox.layers.13.post_attention_layernorm.bias",
    transformer.h.13.attn.bias = "gpt_neox.layers.13.attention.bias",
    transformer.h.13.attn.masked_bias = "gpt_neox.layers.13.attention.masked_bias",
    transformer.h.13.attn.rotary.inv_freq = "gpt_neox.layers.13.attention.rotary_emb.inv_freq",
    transformer.h.13.attn.c_attn.weight = "gpt_neox.layers.13.attention.query_key_value.weight",
    transformer.h.13.attn.c_attn.bias = "gpt_neox.layers.13.attention.query_key_value.bias",
    transformer.h.13.attn.c_proj.weight = "gpt_neox.layers.13.attention.dense.weight",
    transformer.h.13.attn.c_proj.bias = "gpt_neox.layers.13.attention.dense.bias",
    transformer.h.13.mlp.d_1.weight = "gpt_neox.layers.13.mlp.dense_h_to_4h.weight",
    transformer.h.13.mlp.d_1.bias = "gpt_neox.layers.13.mlp.dense_h_to_4h.bias",
    transformer.h.13.mlp.d_2.weight = "gpt_neox.layers.13.mlp.dense_4h_to_h.weight",
    transformer.h.13.mlp.d_2.bias = "gpt_neox.layers.13.mlp.dense_4h_to_h.bias",
    transformer.h.14.ln_1.weight = "gpt_neox.layers.14.input_layernorm.weight",
    transformer.h.14.ln_1.bias = "gpt_neox.layers.14.input_layernorm.bias",
    transformer.h.14.ln_2.weight = "gpt_neox.layers.14.post_attention_layernorm.weight",
    transformer.h.14.ln_2.bias = "gpt_neox.layers.14.post_attention_layernorm.bias",
    transformer.h.14.attn.bias = "gpt_neox.layers.14.attention.bias",
    transformer.h.14.attn.masked_bias = "gpt_neox.layers.14.attention.masked_bias",
    transformer.h.14.attn.rotary.inv_freq = "gpt_neox.layers.14.attention.rotary_emb.inv_freq",
    transformer.h.14.attn.c_attn.weight = "gpt_neox.layers.14.attention.query_key_value.weight",
    transformer.h.14.attn.c_attn.bias = "gpt_neox.layers.14.attention.query_key_value.bias",
    transformer.h.14.attn.c_proj.weight = "gpt_neox.layers.14.attention.dense.weight",
    transformer.h.14.attn.c_proj.bias = "gpt_neox.layers.14.attention.dense.bias",
    transformer.h.14.mlp.d_1.weight = "gpt_neox.layers.14.mlp.dense_h_to_4h.weight",
    transformer.h.14.mlp.d_1.bias = "gpt_neox.layers.14.mlp.dense_h_to_4h.bias",
    transformer.h.14.mlp.d_2.weight = "gpt_neox.layers.14.mlp.dense_4h_to_h.weight",
    transformer.h.14.mlp.d_2.bias = "gpt_neox.layers.14.mlp.dense_4h_to_h.bias",
    transformer.h.15.ln_1.weight = "gpt_neox.layers.15.input_layernorm.weight",
    transformer.h.15.ln_1.bias = "gpt_neox.layers.15.input_layernorm.bias",
    transformer.h.15.ln_2.weight = "gpt_neox.layers.15.post_attention_layernorm.weight",
    transformer.h.15.ln_2.bias = "gpt_neox.layers.15.post_attention_layernorm.bias",
    transformer.h.15.attn.bias = "gpt_neox.layers.15.attention.bias",
    transformer.h.15.attn.masked_bias = "gpt_neox.layers.15.attention.masked_bias",
    transformer.h.15.attn.rotary.inv_freq = "gpt_neox.layers.15.attention.rotary_emb.inv_freq",
    transformer.h.15.attn.c_attn.weight = "gpt_neox.layers.15.attention.query_key_value.weight",
    transformer.h.15.attn.c_attn.bias = "gpt_neox.layers.15.attention.query_key_value.bias",
    transformer.h.15.attn.c_proj.weight = "gpt_neox.layers.15.attention.dense.weight",
    transformer.h.15.attn.c_proj.bias = "gpt_neox.layers.15.attention.dense.bias",
    transformer.h.15.mlp.d_1.weight = "gpt_neox.layers.15.mlp.dense_h_to_4h.weight",
    transformer.h.15.mlp.d_1.bias = "gpt_neox.layers.15.mlp.dense_h_to_4h.bias",
    transformer.h.15.mlp.d_2.weight = "gpt_neox.layers.15.mlp.dense_4h_to_h.weight",
    transformer.h.15.mlp.d_2.bias = "gpt_neox.layers.15.mlp.dense_4h_to_h.bias",
    transformer.ln_f.weight = "gpt_neox.final_layer_norm.weight",
    transformer.ln_f.bias = "gpt_neox.final_layer_norm.bias",
    lm_head.weight = "embed_out.weight",
    transformer.wte.weight = "gpt_neox.embed_in.weight",
    transformer.h.0.ln_1.weight = "gpt_neox.layers.0.input_layernorm.weight",
    transformer.h.0.ln_1.bias = "gpt_neox.layers.0.input_layernorm.bias",
    transformer.h.0.ln_2.weight = "gpt_neox.layers.0.post_attention_layernorm.weight",
    transformer.h.0.ln_2.bias = "gpt_neox.layers.0.post_attention_layernorm.bias",
    transformer.h.0.attn.bias = "gpt_neox.layers.0.attention.bias",
    transformer.h.0.attn.masked_bias = "gpt_neox.layers.0.attention.masked_bias",
    transformer.h.0.attn.rotary.inv_freq = "gpt_neox.layers.0.attention.rotary_emb.inv_freq",
    transformer.h.0.attn.c_attn.weight = "gpt_neox.layers.0.attention.query_key_value.weight",
    transformer.h.0.attn.c_attn.bias = "gpt_neox.layers.0.attention.query_key_value.bias",
    transformer.h.0.attn.c_proj.weight = "gpt_neox.layers.0.attention.dense.weight",
    transformer.h.0.attn.c_proj.bias = "gpt_neox.layers.0.attention.dense.bias",
    transformer.h.0.mlp.d_1.weight = "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
    transformer.h.0.mlp.d_1.bias = "gpt_neox.layers.0.mlp.dense_h_to_4h.bias",
    transformer.h.0.mlp.d_2.weight = "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
    transformer.h.0.mlp.d_2.bias = "gpt_neox.layers.0.mlp.dense_4h_to_h.bias",
    transformer.h.1.ln_1.weight = "gpt_neox.layers.1.input_layernorm.weight",
    transformer.h.1.ln_1.bias = "gpt_neox.layers.1.input_layernorm.bias",
    transformer.h.1.ln_2.weight = "gpt_neox.layers.1.post_attention_layernorm.weight",
    transformer.h.1.ln_2.bias = "gpt_neox.layers.1.post_attention_layernorm.bias",
    transformer.h.1.attn.bias = "gpt_neox.layers.1.attention.bias",
    transformer.h.1.attn.masked_bias = "gpt_neox.layers.1.attention.masked_bias",
    transformer.h.1.attn.rotary.inv_freq = "gpt_neox.layers.1.attention.rotary_emb.inv_freq",
    transformer.h.1.attn.c_attn.weight = "gpt_neox.layers.1.attention.query_key_value.weight",
    transformer.h.1.attn.c_attn.bias = "gpt_neox.layers.1.attention.query_key_value.bias",
    transformer.h.1.attn.c_proj.weight = "gpt_neox.layers.1.attention.dense.weight",
    transformer.h.1.attn.c_proj.bias = "gpt_neox.layers.1.attention.dense.bias",
    transformer.h.1.mlp.d_1.weight = "gpt_neox.layers.1.mlp.dense_h_to_4h.weight",
    transformer.h.1.mlp.d_1.bias = "gpt_neox.layers.1.mlp.dense_h_to_4h.bias",
    transformer.h.1.mlp.d_2.weight = "gpt_neox.layers.1.mlp.dense_4h_to_h.weight",
    transformer.h.1.mlp.d_2.bias = "gpt_neox.layers.1.mlp.dense_4h_to_h.bias",
    transformer.h.2.ln_1.weight = "gpt_neox.layers.2.input_layernorm.weight",
    transformer.h.2.ln_1.bias = "gpt_neox.layers.2.input_layernorm.bias",
    transformer.h.2.ln_2.weight = "gpt_neox.layers.2.post_attention_layernorm.weight",
    transformer.h.2.ln_2.bias = "gpt_neox.layers.2.post_attention_layernorm.bias",
    transformer.h.2.attn.bias = "gpt_neox.layers.2.attention.bias",
    transformer.h.2.attn.masked_bias = "gpt_neox.layers.2.attention.masked_bias",
    transformer.h.2.attn.rotary.inv_freq = "gpt_neox.layers.2.attention.rotary_emb.inv_freq",
    transformer.h.2.attn.c_attn.weight = "gpt_neox.layers.2.attention.query_key_value.weight",
    transformer.h.2.attn.c_attn.bias = "gpt_neox.layers.2.attention.query_key_value.bias",
    transformer.h.2.attn.c_proj.weight = "gpt_neox.layers.2.attention.dense.weight",
    transformer.h.2.attn.c_proj.bias = "gpt_neox.layers.2.attention.dense.bias",
    transformer.h.2.mlp.d_1.weight = "gpt_neox.layers.2.mlp.dense_h_to_4h.weight",
    transformer.h.2.mlp.d_1.bias = "gpt_neox.layers.2.mlp.dense_h_to_4h.bias",
    transformer.h.2.mlp.d_2.weight = "gpt_neox.layers.2.mlp.dense_4h_to_h.weight",
    transformer.h.2.mlp.d_2.bias = "gpt_neox.layers.2.mlp.dense_4h_to_h.bias",
    transformer.h.3.ln_1.weight = "gpt_neox.layers.3.input_layernorm.weight",
    transformer.h.3.ln_1.bias = "gpt_neox.layers.3.input_layernorm.bias",
    transformer.h.3.ln_2.weight = "gpt_neox.layers.3.post_attention_layernorm.weight",
    transformer.h.3.ln_2.bias = "gpt_neox.layers.3.post_attention_layernorm.bias",
    transformer.h.3.attn.bias = "gpt_neox.layers.3.attention.bias",
    transformer.h.3.attn.masked_bias = "gpt_neox.layers.3.attention.masked_bias",
    transformer.h.3.attn.rotary.inv_freq = "gpt_neox.layers.3.attention.rotary_emb.inv_freq",
    transformer.h.3.attn.c_attn.weight = "gpt_neox.layers.3.attention.query_key_value.weight",
    transformer.h.3.attn.c_attn.bias = "gpt_neox.layers.3.attention.query_key_value.bias",
    transformer.h.3.attn.c_proj.weight = "gpt_neox.layers.3.attention.dense.weight",
    transformer.h.3.attn.c_proj.bias = "gpt_neox.layers.3.attention.dense.bias",
    transformer.h.3.mlp.d_1.weight = "gpt_neox.layers.3.mlp.dense_h_to_4h.weight",
    transformer.h.3.mlp.d_1.bias = "gpt_neox.layers.3.mlp.dense_h_to_4h.bias",
    transformer.h.3.mlp.d_2.weight = "gpt_neox.layers.3.mlp.dense_4h_to_h.weight",
    transformer.h.3.mlp.d_2.bias = "gpt_neox.layers.3.mlp.dense_4h_to_h.bias",
    transformer.h.4.ln_1.weight = "gpt_neox.layers.4.input_layernorm.weight",
    transformer.h.4.ln_1.bias = "gpt_neox.layers.4.input_layernorm.bias",
    transformer.h.4.ln_2.weight = "gpt_neox.layers.4.post_attention_layernorm.weight",
    transformer.h.4.ln_2.bias = "gpt_neox.layers.4.post_attention_layernorm.bias",
    transformer.h.4.attn.bias = "gpt_neox.layers.4.attention.bias",
    transformer.h.4.attn.masked_bias = "gpt_neox.layers.4.attention.masked_bias",
    transformer.h.4.attn.rotary.inv_freq = "gpt_neox.layers.4.attention.rotary_emb.inv_freq",
    transformer.h.4.attn.c_attn.weight = "gpt_neox.layers.4.attention.query_key_value.weight",
    transformer.h.4.attn.c_attn.bias = "gpt_neox.layers.4.attention.query_key_value.bias",
    transformer.h.4.attn.c_proj.weight = "gpt_neox.layers.4.attention.dense.weight",
    transformer.h.4.attn.c_proj.bias = "gpt_neox.layers.4.attention.dense.bias",
    transformer.h.4.mlp.d_1.weight = "gpt_neox.layers.4.mlp.dense_h_to_4h.weight",
    transformer.h.4.mlp.d_1.bias = "gpt_neox.layers.4.mlp.dense_h_to_4h.bias",
    transformer.h.4.mlp.d_2.weight = "gpt_neox.layers.4.mlp.dense_4h_to_h.weight",
    transformer.h.4.mlp.d_2.bias = "gpt_neox.layers.4.mlp.dense_4h_to_h.bias",
    transformer.h.5.ln_1.weight = "gpt_neox.layers.5.input_layernorm.weight",
    transformer.h.5.ln_1.bias = "gpt_neox.layers.5.input_layernorm.bias",
    transformer.h.5.ln_2.weight = "gpt_neox.layers.5.post_attention_layernorm.weight",
    transformer.h.5.ln_2.bias = "gpt_neox.layers.5.post_attention_layernorm.bias",
    transformer.h.5.attn.bias = "gpt_neox.layers.5.attention.bias",
    transformer.h.5.attn.masked_bias = "gpt_neox.layers.5.attention.masked_bias",
    transformer.h.5.attn.rotary.inv_freq = "gpt_neox.layers.5.attention.rotary_emb.inv_freq",
    transformer.h.5.attn.c_attn.weight = "gpt_neox.layers.5.attention.query_key_value.weight",
    transformer.h.5.attn.c_attn.bias = "gpt_neox.layers.5.attention.query_key_value.bias",
    transformer.h.5.attn.c_proj.weight = "gpt_neox.layers.5.attention.dense.weight",
    transformer.h.5.attn.c_proj.bias = "gpt_neox.layers.5.attention.dense.bias",
    transformer.h.5.mlp.d_1.weight = "gpt_neox.layers.5.mlp.dense_h_to_4h.weight",
    transformer.h.5.mlp.d_1.bias = "gpt_neox.layers.5.mlp.dense_h_to_4h.bias",
    transformer.h.5.mlp.d_2.weight = "gpt_neox.layers.5.mlp.dense_4h_to_h.weight",
    transformer.h.5.mlp.d_2.bias = "gpt_neox.layers.5.mlp.dense_4h_to_h.bias",
    transformer.h.6.ln_1.weight = "gpt_neox.layers.6.input_layernorm.weight",
    transformer.h.6.ln_1.bias = "gpt_neox.layers.6.input_layernorm.bias",
    transformer.h.6.ln_2.weight = "gpt_neox.layers.6.post_attention_layernorm.weight",
    transformer.h.6.ln_2.bias = "gpt_neox.layers.6.post_attention_layernorm.bias",
    transformer.h.6.attn.bias = "gpt_neox.layers.6.attention.bias",
    transformer.h.6.attn.masked_bias = "gpt_neox.layers.6.attention.masked_bias",
    transformer.h.6.attn.rotary.inv_freq = "gpt_neox.layers.6.attention.rotary_emb.inv_freq",
    transformer.h.6.attn.c_attn.weight = "gpt_neox.layers.6.attention.query_key_value.weight",
    transformer.h.6.attn.c_attn.bias = "gpt_neox.layers.6.attention.query_key_value.bias",
    transformer.h.6.attn.c_proj.weight = "gpt_neox.layers.6.attention.dense.weight",
    transformer.h.6.attn.c_proj.bias = "gpt_neox.layers.6.attention.dense.bias",
    transformer.h.6.mlp.d_1.weight = "gpt_neox.layers.6.mlp.dense_h_to_4h.weight",
    transformer.h.6.mlp.d_1.bias = "gpt_neox.layers.6.mlp.dense_h_to_4h.bias",
    transformer.h.6.mlp.d_2.weight = "gpt_neox.layers.6.mlp.dense_4h_to_h.weight",
    transformer.h.6.mlp.d_2.bias = "gpt_neox.layers.6.mlp.dense_4h_to_h.bias",
    transformer.h.7.ln_1.weight = "gpt_neox.layers.7.input_layernorm.weight",
    transformer.h.7.ln_1.bias = "gpt_neox.layers.7.input_layernorm.bias",
    transformer.h.7.ln_2.weight = "gpt_neox.layers.7.post_attention_layernorm.weight",
    transformer.h.7.ln_2.bias = "gpt_neox.layers.7.post_attention_layernorm.bias",
    transformer.h.7.attn.bias = "gpt_neox.layers.7.attention.bias",
    transformer.h.7.attn.masked_bias = "gpt_neox.layers.7.attention.masked_bias",
    transformer.h.7.attn.rotary.inv_freq = "gpt_neox.layers.7.attention.rotary_emb.inv_freq",
    transformer.h.7.attn.c_attn.weight = "gpt_neox.layers.7.attention.query_key_value.weight",
    transformer.h.7.attn.c_attn.bias = "gpt_neox.layers.7.attention.query_key_value.bias",
    transformer.h.7.attn.c_proj.weight = "gpt_neox.layers.7.attention.dense.weight",
    transformer.h.7.attn.c_proj.bias = "gpt_neox.layers.7.attention.dense.bias",
    transformer.h.7.mlp.d_1.weight = "gpt_neox.layers.7.mlp.dense_h_to_4h.weight",
    transformer.h.7.mlp.d_1.bias = "gpt_neox.layers.7.mlp.dense_h_to_4h.bias",
    transformer.h.7.mlp.d_2.weight = "gpt_neox.layers.7.mlp.dense_4h_to_h.weight",
    transformer.h.7.mlp.d_2.bias = "gpt_neox.layers.7.mlp.dense_4h_to_h.bias",
    transformer.h.8.ln_1.weight = "gpt_neox.layers.8.input_layernorm.weight",
    transformer.h.8.ln_1.bias = "gpt_neox.layers.8.input_layernorm.bias",
    transformer.h.8.ln_2.weight = "gpt_neox.layers.8.post_attention_layernorm.weight",
    transformer.h.8.ln_2.bias = "gpt_neox.layers.8.post_attention_layernorm.bias",
    transformer.h.8.attn.bias = "gpt_neox.layers.8.attention.bias",
    transformer.h.8.attn.masked_bias = "gpt_neox.layers.8.attention.masked_bias",
    transformer.h.8.attn.rotary.inv_freq = "gpt_neox.layers.8.attention.rotary_emb.inv_freq",
    transformer.h.8.attn.c_attn.weight = "gpt_neox.layers.8.attention.query_key_value.weight",
    transformer.h.8.attn.c_attn.bias = "gpt_neox.layers.8.attention.query_key_value.bias",
    transformer.h.8.attn.c_proj.weight = "gpt_neox.layers.8.attention.dense.weight",
    transformer.h.8.attn.c_proj.bias = "gpt_neox.layers.8.attention.dense.bias",
    transformer.h.8.mlp.d_1.weight = "gpt_neox.layers.8.mlp.dense_h_to_4h.weight",
    transformer.h.8.mlp.d_1.bias = "gpt_neox.layers.8.mlp.dense_h_to_4h.bias",
    transformer.h.8.mlp.d_2.weight = "gpt_neox.layers.8.mlp.dense_4h_to_h.weight",
    transformer.h.8.mlp.d_2.bias = "gpt_neox.layers.8.mlp.dense_4h_to_h.bias",
    transformer.h.9.ln_1.weight = "gpt_neox.layers.9.input_layernorm.weight",
    transformer.h.9.ln_1.bias = "gpt_neox.layers.9.input_layernorm.bias",
    transformer.h.9.ln_2.weight = "gpt_neox.layers.9.post_attention_layernorm.weight",
    transformer.h.9.ln_2.bias = "gpt_neox.layers.9.post_attention_layernorm.bias",
    transformer.h.9.attn.bias = "gpt_neox.layers.9.attention.bias",
    transformer.h.9.attn.masked_bias = "gpt_neox.layers.9.attention.masked_bias",
    transformer.h.9.attn.rotary.inv_freq = "gpt_neox.layers.9.attention.rotary_emb.inv_freq",
    transformer.h.9.attn.c_attn.weight = "gpt_neox.layers.9.attention.query_key_value.weight",
    transformer.h.9.attn.c_attn.bias = "gpt_neox.layers.9.attention.query_key_value.bias",
    transformer.h.9.attn.c_proj.weight = "gpt_neox.layers.9.attention.dense.weight",
    transformer.h.9.attn.c_proj.bias = "gpt_neox.layers.9.attention.dense.bias",
    transformer.h.9.mlp.d_1.weight = "gpt_neox.layers.9.mlp.dense_h_to_4h.weight",
    transformer.h.9.mlp.d_1.bias = "gpt_neox.layers.9.mlp.dense_h_to_4h.bias",
    transformer.h.9.mlp.d_2.weight = "gpt_neox.layers.9.mlp.dense_4h_to_h.weight",
    transformer.h.9.mlp.d_2.bias = "gpt_neox.layers.9.mlp.dense_4h_to_h.bias",
    transformer.h.10.ln_1.weight = "gpt_neox.layers.10.input_layernorm.weight",
    transformer.h.10.ln_1.bias = "gpt_neox.layers.10.input_layernorm.bias",
    transformer.h.10.ln_2.weight = "gpt_neox.layers.10.post_attention_layernorm.weight",
    transformer.h.10.ln_2.bias = "gpt_neox.layers.10.post_attention_layernorm.bias",
    transformer.h.10.attn.bias = "gpt_neox.layers.10.attention.bias",
    transformer.h.10.attn.masked_bias = "gpt_neox.layers.10.attention.masked_bias",
    transformer.h.10.attn.rotary.inv_freq = "gpt_neox.layers.10.attention.rotary_emb.inv_freq",
    transformer.h.10.attn.c_attn.weight = "gpt_neox.layers.10.attention.query_key_value.weight",
    transformer.h.10.attn.c_attn.bias = "gpt_neox.layers.10.attention.query_key_value.bias",
    transformer.h.10.attn.c_proj.weight = "gpt_neox.layers.10.attention.dense.weight",
    transformer.h.10.attn.c_proj.bias = "gpt_neox.layers.10.attention.dense.bias",
    transformer.h.10.mlp.d_1.weight = "gpt_neox.layers.10.mlp.dense_h_to_4h.weight",
    transformer.h.10.mlp.d_1.bias = "gpt_neox.layers.10.mlp.dense_h_to_4h.bias",
    transformer.h.10.mlp.d_2.weight = "gpt_neox.layers.10.mlp.dense_4h_to_h.weight",
    transformer.h.10.mlp.d_2.bias = "gpt_neox.layers.10.mlp.dense_4h_to_h.bias",
    transformer.h.11.ln_1.weight = "gpt_neox.layers.11.input_layernorm.weight",
    transformer.h.11.ln_1.bias = "gpt_neox.layers.11.input_layernorm.bias",
    transformer.h.11.ln_2.weight = "gpt_neox.layers.11.post_attention_layernorm.weight",
    transformer.h.11.ln_2.bias = "gpt_neox.layers.11.post_attention_layernorm.bias",
    transformer.h.11.attn.bias = "gpt_neox.layers.11.attention.bias",
    transformer.h.11.attn.masked_bias = "gpt_neox.layers.11.attention.masked_bias",
    transformer.h.11.attn.rotary.inv_freq = "gpt_neox.layers.11.attention.rotary_emb.inv_freq",
    transformer.h.11.attn.c_attn.weight = "gpt_neox.layers.11.attention.query_key_value.weight",
    transformer.h.11.attn.c_attn.bias = "gpt_neox.layers.11.attention.query_key_value.bias",
    transformer.h.11.attn.c_proj.weight = "gpt_neox.layers.11.attention.dense.weight",
    transformer.h.11.attn.c_proj.bias = "gpt_neox.layers.11.attention.dense.bias"
  )
}

