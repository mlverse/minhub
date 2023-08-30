# References:
# - https://github.com/karpathy/minGPT
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py

#' @noRd
#' @importFrom zeallot %<-%
#' @importFrom purrr map
#' @import torch
NULL

nn_llama_rmsnorm <- nn_module(
  initialize = function(hidden_size, eps = 1e-6) {
    self$weight <- nn_parameter(torch_ones(hidden_size))
    self$eps <- eps
  },
  forward = function(x) {
    dtype <- x$dtype

    variance <- x$
      to(dtype="float32")$
      pow(2)$
      mean(-1, keepdim=TRUE)

    x <- x * torch_rsqrt(variance + self$eps)
    (self$weight * x)$to(dtype=dtype)
  }
)

llama_rotate_half <- function(x) {
  c(x1, x2) %<-% torch_split(x, x$size(-1) / 2, -1)
  torch_cat(list(-x2, x1), dim = -1)
}

nn_llama_rotary_embedding <- nn_module(
  initialize = function(n_embd, max_pos, base=10000) {
    self$n_embd <- n_embd
    self$max_pos <- max_pos
    self$base <- base

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

      self$inv_freq <- nn_buffer(
        torch_ones(1) / (self$base ^ (torch_arange(0, self$n_embd-1, step = 2) / self$n_embd)),
        persistent = FALSE # unlike llama this is not a persistent buffer
      )

      freqs <- torch_arange(start = 0, end = self$max_pos - 1)$
        float()$
        outer(self$inv_freq)$
        view(c(1,1, self$max_pos, self$n_embd/2))

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
    x * cos + llama_rotate_half(x) * sin
  }
)

nn_llama_attention <- nn_module(
  initialize = function(n_head, n_embd, max_pos, rope_base) {
    self$n_head <- n_head
    self$n_embd <- n_embd

    self$max_pos <- max_pos
    self$rotary <- nn_llama_rotary_embedding(n_embd/n_head, max_pos, rope_base)

    self$q_proj <- nn_linear(n_embd, n_embd, bias = FALSE)
    self$k_proj <- nn_linear(n_embd, n_embd, bias = FALSE)
    self$v_proj <- nn_linear(n_embd, n_embd, bias = FALSE)
    self$o_proj <- nn_linear(n_embd, n_embd, bias = FALSE)

    self$cached_bias()
  },
  forward = function(x) {
    c(b, t, h) %<-% x$shape

    # (b, t, h) -> [(b, nh, t, h/nh) * 3]
    q <- self$q_proj(x)$view(c(b, t, self$n_head, -1))$transpose(2, 3)
    k <- self$k_proj(x)$view(c(b, t, self$n_head, -1))$transpose(2, 3)
    v <- self$v_proj(x)$view(c(b, t, self$n_head, -1))$transpose(2, 3)

    q <- self$rotary(q)$to(dtype="float")
    k <- self$rotary(k)$to(dtype="float")

    # the following block requires key and value to be in float32 otherwise
    # it leads to precision problems
    att <- torch_matmul(q, k$transpose(-2, -1)) * (1 / sqrt(k$size(-1)))
    att <- att$masked_fill(self$bias[,,1:t, 1:t] == 0, self$masked_bias)
    att <- nnf_softmax(att, dim=-1)$to(dtype = v$dtype)

    y <- torch_matmul(att, v)$transpose(2, 3)$contiguous()$view(c(b, t, h))
    self$o_proj(y)
  },
  .load_from_state_dict = function(...) {
    super$.load_from_state_dict(...)
    self$cached_bias()
  },
  cached_bias = function() {
    # causal mask to ensure that attention is only applied to the left in the
    # input sequence
    self$bias <- torch_ones(self$max_pos, self$max_pos)$
      bool()$
      tril()$
      view(c(1, 1, self$max_pos, self$max_pos)) |>
      nn_buffer(persistent = FALSE)

    self$masked_bias <- nn_buffer(torch_scalar_tensor(-Inf), persistent = FALSE)
  }
)

nn_llama_mlp <- nn_module(
  initialize = function(n_embd, n_inter) {
    self$gate_proj <- nn_linear(n_embd, n_inter, bias = FALSE)
    self$down_proj <- nn_linear(n_inter, n_embd, bias = FALSE)
    self$up_proj <- nn_linear(n_embd, n_inter, bias = FALSE)
    self$act <- nn_silu()
  },
  forward = function(x) {
    self$down_proj(self$act(self$gate_proj(x)) * self$up_proj(x))
  }
)

nn_llama_layer <- nn_module(
  initialize = function(n_embd, n_inter, n_head, max_pos, rmsnorm_eps, rope_base) {
    self$ln_1 <- nn_llama_rmsnorm(n_embd, rmsnorm_eps)
    self$ln_2 <- nn_llama_rmsnorm(n_embd, rmsnorm_eps)
    self$attn <- nn_llama_attention(n_head, n_embd, max_pos, rope_base)
    self$mlp <- nn_llama_mlp(n_embd, n_inter)
  },
  forward = function(x) {
    x <- x + self$attn(self$ln_1(x))
    x + self$mlp(self$ln_2(x))
  }
)

nn_llama_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_inter, n_head, n_layer, max_pos,
                        rmsnorm_eps, rope_base) {
    self$transformer <- nn_module_dict(list(
      wte = nn_embedding(vocab_size, n_embd),
      h = nn_sequential(!!!map(
        1:n_layer,
        \(x) nn_llama_layer(n_embd, n_inter, n_head, max_pos, rmsnorm_eps, rope_base)
      )),
      ln_f = nn_llama_rmsnorm(n_embd, rmsnorm_eps)
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

#' llama
#'
#' Initializes a llama like model
#'
#' @param vocab_size An integer indicating the size of the vocabulary or the number
#'   of unique tokens in the input data.
#' @param n_embd An integer specifying the dimensionality of the embedding vectors.
#' @param n_inter An integer specifying the size of the intermediate layer in the MLP
#' @param n_head An integer representing the number of attention heads in the
#'   multi-head attention mechanism.
#' @param n_layer An integer indicating the number of layers in the deep learning model.
#' @param max_pos An integer specifying the maximum position encoding value or
#'  the maximum sequence length.
#' @param rmsnorm_eps The epsilon used by the rms normalization layers.
#' @param rope_base The base period of the RoPE embeddings.
#' @param identifier A string representing the identifier or name of the pre-trained
#'  model in the Hugging Face model hub.
#' @param revision A string specifying the revision or version of the pre-trained
#'  model in the Hugging Face model hub.
#' @returns An initialized [torch::nn_module()].
#' @export
llama <- function(vocab_size=50432, n_embd=6144, n_inter = 11008, n_head=64,
                  n_layer=44, max_pos=2048, rmsnorm_eps = 1e-6, rope_base = 10000) {
  nn_llama_model(vocab_size, n_embd, n_inter, n_head, n_layer, max_pos, rmsnorm_eps,
                 rope_base)
}

#' @describeIn llama Initializes a llama model using a configuration defined in HF Hub
#' @export
llama_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  if (config$model_type != "llama")
    cli::cli_abort(c(
      "{.arg config$model_type} must be {.val llama}, got {.val {config$model_type}}"
    ))

  if (config$hidden_act != "silu")
    cli::cli_abort(c(
      x = "Unsupported {.arg config$hidden_act}: {.val {config$hidden_act}}",
      i = "Currently only {.val silu} is supported."
    ))

  # remap HF config attributes to minhub configurations
  vocab_size  <- config$vocab_size
  n_embd      <- config$hidden_size
  n_inter     <- config$intermediate_size
  n_head      <- config$num_attention_heads
  n_layer     <- config$num_hidden_layers
  max_pos     <- config$max_position_embeddings
  rmsnorm_eps <- config$rms_norm_eps # unlike most llama models, we also havve different values for this
  rope_base   <- config$rope_theta %||% 10000 # unlike LLama, code llama models tune this

  llama(vocab_size, n_embd, n_inter, n_head, n_layer, max_pos, rmsnorm_eps, rope_base)
}

#' @describeIn llama Initializes the llama model and load pre-trained weights from HF hub.
#' @export
llama_from_pretrained <- function(identifier, revision = "main") {
  with_device(device="meta", {
    model <- llama_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- llama_hf_weights_remap(state_dict)

  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

llama_hf_weights_remap <- function(state_dict) {
  nms <- names(state_dict)
  nms <- gsub("model.embed_tokens.weight", "transformer.wte.weight", nms, fixed = TRUE)
  nms <- gsub("model.layers", "transformer.h", nms, fixed = TRUE)
  nms <- gsub("self_attn", "attn", nms, fixed = TRUE)
  nms <- gsub("input_layernorm", "ln_1", nms, fixed = TRUE)
  nms <- gsub("post_attention_layernorm", "ln_2", nms, fixed = TRUE)
  nms <- gsub("model.norm", "transformer.ln_f", nms, fixed = TRUE)
  nms <- gsub("rotary_emb", "rotary", nms, fixed = TRUE)
  names(state_dict) <- nms
  state_dict
}

