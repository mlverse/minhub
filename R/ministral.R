# References:
# - https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py

#' @noRd
#' @importFrom zeallot %<-%
#' @importFrom purrr map
#' @import torch
NULL

# YaRN RoPE helper functions
yarn_find_correction_dim <- function(num_rotations, dim, base, max_position_embeddings) {
  (dim * log(max_position_embeddings / (num_rotations * 2 * pi))) / (2 * log(base))
}

yarn_find_correction_range <- function(low_rot, high_rot, dim, base, max_position_embeddings) {
  low <- floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
  high <- ceiling(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
  c(max(low, 0), min(high, dim - 1))
}

yarn_linear_ramp_mask <- function(min_val, max_val, dim, dtype = torch_float32()) {
  if (min_val == max_val) min_val <- min_val - 0.001
  linear_func <- (torch_arange(0, dim - 1, dtype = dtype) - min_val) / (max_val - min_val)
  torch_clamp(linear_func, 0, 1)
}

ministral_rotate_half <- function(x) {
  c(x1, x2) %<-% torch_split(x, x$size(-1) / 2, -1)
  torch_cat(list(-x2, x1), dim = -1)
}

repeat_kv <- function(hidden_states, n_rep) {
  if (n_rep == 1) return(hidden_states)
  c(batch, num_kv_heads, seq_len, head_dim) %<-% hidden_states$shape
  hidden_states$unsqueeze(3)$
    expand(c(batch, num_kv_heads, n_rep, seq_len, head_dim))$
    reshape(c(batch, num_kv_heads * n_rep, seq_len, head_dim))
}

nn_ministral_rmsnorm <- nn_module(
  initialize = function(hidden_size, eps = 1e-5) {
    self$weight <- nn_parameter(torch_ones(hidden_size))
    self$eps <- eps
  },
  forward = function(x) {
    dtype <- x$dtype
    variance <- x$to(dtype = "float32")$pow(2)$mean(-1, keepdim = TRUE)
    x <- x * torch_rsqrt(variance + self$eps)
    (self$weight * x)$to(dtype = dtype)
  }
)

nn_ministral_yarn_rotary_embedding <- nn_module(
  initialize = function(head_dim, max_pos, base, factor, beta_fast, beta_slow,
                        original_max_pos, mscale, mscale_all_dim) {
    self$head_dim <- head_dim
    self$max_pos <- max_pos
    self$base <- base
    self$factor <- factor
    self$beta_fast <- beta_fast
    self$beta_slow <- beta_slow
    self$original_max_pos <- original_max_pos
    self$mscale <- mscale
    self$mscale_all_dim <- mscale_all_dim
    self$cached_embeddings()
  },
  .load_from_state_dict = function(...) {
    super$.load_from_state_dict(...)
    self$cached_embeddings(invalidate = TRUE)
  },
 get_mscale = function(scale, mscale) {
    if (mscale <= 0) return(1.0)
    0.1 * mscale * log(scale) + 1.0
  },
  cached_embeddings = function(t = 1, invalidate = FALSE) {
    invalidate <- invalidate || is.null(self$cos)
    if (invalidate) {
      dim <- self$head_dim
      pos_freqs <- self$base ^ (torch_arange(0, dim - 1, step = 2) / dim)
      inv_freq_extrapolation <- 1.0 / pos_freqs
      inv_freq_interpolation <- 1.0 / (self$factor * pos_freqs)

      c(low, high) %<-% yarn_find_correction_range(
        self$beta_slow, self$beta_fast, dim, self$base, self$original_max_pos
      )
      inv_freq_extrapolation_factor <- yarn_linear_ramp_mask(low, high, dim / 2)

      inv_freq <- inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) +
                  inv_freq_extrapolation * inv_freq_extrapolation_factor
      self$inv_freq <- nn_buffer(inv_freq, persistent = FALSE)

      self$attention_scale <- self$get_mscale(self$factor, self$mscale) /
                              self$get_mscale(self$factor, self$mscale_all_dim)

      freqs <- torch_arange(start = 0, end = self$max_pos - 1)$
        float()$outer(self$inv_freq)$view(c(1, 1, self$max_pos, dim / 2))
      emb <- torch_cat(list(freqs, freqs), dim = -1)
      self$cos <- nn_buffer(emb$cos(), persistent = FALSE)
      self$sin <- nn_buffer(emb$sin(), persistent = FALSE)
    }
    list(self$cos[,,1:t,], self$sin[,,1:t,], self$attention_scale)
  },
  forward = function(x) {
    c(b, nh, t, ed) %<-% x$shape
    c(cos, sin, attn_scale) %<-% self$cached_embeddings(t)
    (x * cos + ministral_rotate_half(x) * sin) * attn_scale
  }
)

nn_ministral_attention <- nn_module(
  initialize = function(n_embd, n_head, n_kv_head, head_dim, max_pos,
                        rope_base, rope_factor, rope_beta_fast, rope_beta_slow,
                        rope_original_max_pos, rope_mscale, rope_mscale_all_dim) {
    self$n_head <- n_head
    self$n_kv_head <- n_kv_head
    self$head_dim <- head_dim
    self$n_kv_groups <- n_head %/% n_kv_head
    self$max_pos <- max_pos

    self$rotary <- nn_ministral_yarn_rotary_embedding(
      head_dim, max_pos, rope_base, rope_factor, rope_beta_fast, rope_beta_slow,
      rope_original_max_pos, rope_mscale, rope_mscale_all_dim
    )

    self$q_proj <- nn_linear(n_embd, n_head * head_dim, bias = FALSE)
    self$k_proj <- nn_linear(n_embd, n_kv_head * head_dim, bias = FALSE)
    self$v_proj <- nn_linear(n_embd, n_kv_head * head_dim, bias = FALSE)
    self$o_proj <- nn_linear(n_head * head_dim, n_embd, bias = FALSE)
    self$cached_bias()
  },
  forward = function(x) {
    c(b, t, h) %<-% x$shape

    q <- self$q_proj(x)$view(c(b, t, self$n_head, self$head_dim))$transpose(2, 3)
    k <- self$k_proj(x)$view(c(b, t, self$n_kv_head, self$head_dim))$transpose(2, 3)
    v <- self$v_proj(x)$view(c(b, t, self$n_kv_head, self$head_dim))$transpose(2, 3)

    q <- self$rotary(q)$to(dtype = "float")
    k <- self$rotary(k)$to(dtype = "float")

    k <- repeat_kv(k, self$n_kv_groups)
    v <- repeat_kv(v, self$n_kv_groups)

    att <- torch_matmul(q, k$transpose(-2, -1)) * (1 / sqrt(self$head_dim))
    att <- att$masked_fill(self$bias[,,1:t, 1:t] == 0, self$masked_bias)
    att <- nnf_softmax(att, dim = -1)$to(dtype = v$dtype)

    y <- torch_matmul(att, v)$transpose(2, 3)$contiguous()$
      view(c(b, t, self$n_head * self$head_dim))
    self$o_proj(y)
  },
  .load_from_state_dict = function(...) {
    super$.load_from_state_dict(...)
    self$cached_bias()
  },
  cached_bias = function() {
    self$bias <- torch_ones(self$max_pos, self$max_pos)$bool()$tril()$
      view(c(1, 1, self$max_pos, self$max_pos)) |> nn_buffer(persistent = FALSE)
    self$masked_bias <- nn_buffer(torch_scalar_tensor(-Inf), persistent = FALSE)
  }
)

nn_ministral_mlp <- nn_module(
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

nn_ministral_layer <- nn_module(
  initialize = function(n_embd, n_inter, n_head, n_kv_head, head_dim, max_pos,
                        rmsnorm_eps, rope_base, rope_factor, rope_beta_fast,
                        rope_beta_slow, rope_original_max_pos, rope_mscale,
                        rope_mscale_all_dim) {
    self$ln_1 <- nn_ministral_rmsnorm(n_embd, rmsnorm_eps)
    self$ln_2 <- nn_ministral_rmsnorm(n_embd, rmsnorm_eps)
    self$attn <- nn_ministral_attention(
      n_embd, n_head, n_kv_head, head_dim, max_pos, rope_base, rope_factor,
      rope_beta_fast, rope_beta_slow, rope_original_max_pos, rope_mscale,
      rope_mscale_all_dim
    )
    self$mlp <- nn_ministral_mlp(n_embd, n_inter)
  },
  forward = function(x) {
    x <- x + self$attn(self$ln_1(x))
    x + self$mlp(self$ln_2(x))
  }
)

nn_ministral_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_inter, n_head, n_kv_head, head_dim,
                        n_layer, max_pos, rmsnorm_eps, rope_base, rope_factor,
                        rope_beta_fast, rope_beta_slow, rope_original_max_pos,
                        rope_mscale, rope_mscale_all_dim) {
    self$transformer <- nn_module_dict(list(
      wte = nn_embedding(vocab_size, n_embd),
      h = nn_sequential(!!!map(
        1:n_layer,
        \(x) nn_ministral_layer(
          n_embd, n_inter, n_head, n_kv_head, head_dim, max_pos, rmsnorm_eps,
          rope_base, rope_factor, rope_beta_fast, rope_beta_slow,
          rope_original_max_pos, rope_mscale, rope_mscale_all_dim
        )
      )),
      ln_f = nn_ministral_rmsnorm(n_embd, rmsnorm_eps)
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

#' ministral
#'
#' Initializes a Ministral-like model with YaRN RoPE and GQA
#'
#' @param vocab_size Vocabulary size.
#' @param n_embd Embedding dimension.
#' @param n_inter Intermediate size in MLP.
#' @param n_head Number of attention heads.
#' @param n_kv_head Number of key/value heads (for GQA).
#' @param head_dim Dimension of each attention head.
#' @param n_layer Number of transformer layers.
#' @param max_pos Maximum position embeddings.
#' @param rmsnorm_eps Epsilon for RMSNorm.
#' @param rope_base Base for rotary embeddings.
#' @param rope_factor YaRN scaling factor.
#' @param rope_beta_fast YaRN beta_fast parameter.
#' @param rope_beta_slow YaRN beta_slow parameter.
#' @param rope_original_max_pos Original max position embeddings for YaRN.
#' @param rope_mscale YaRN mscale parameter.
#' @param rope_mscale_all_dim YaRN mscale_all_dim parameter.
#' @param identifier HuggingFace model identifier.
#' @param revision HuggingFace model revision.
#' @returns An initialized [torch::nn_module()].
#' @export
ministral <- function(vocab_size = 131072, n_embd = 5120, n_inter = 16384,
                      n_head = 32, n_kv_head = 8, head_dim = 128, n_layer = 40,
                      max_pos = 262144, rmsnorm_eps = 1e-5, rope_base = 1e9,
                      rope_factor = 16, rope_beta_fast = 32, rope_beta_slow = 1,
                      rope_original_max_pos = 16384, rope_mscale = 1,
                      rope_mscale_all_dim = 1) {
  nn_ministral_model(
    vocab_size, n_embd, n_inter, n_head, n_kv_head, head_dim, n_layer, max_pos,
    rmsnorm_eps, rope_base, rope_factor, rope_beta_fast, rope_beta_slow,
    rope_original_max_pos, rope_mscale, rope_mscale_all_dim
  )
}

#' @describeIn ministral Initializes from HuggingFace config
#' @export
ministral_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  # Handle multimodal config (text_config nested)
  if (!is.null(config$text_config)) {
    rope_params <- config$text_config$rope_parameters
    config <- config$text_config
  } else {
    rope_params <- config$rope_parameters %||% config
  }

  if (!config$model_type %in% c("ministral3", "mistral"))
    cli::cli_abort("Unsupported model_type: {.val {config$model_type}}")

  if (config$hidden_act != "silu")
    cli::cli_abort("Unsupported hidden_act: {.val {config$hidden_act}}")

  ministral(
    vocab_size = config$vocab_size,
    n_embd = config$hidden_size,
    n_inter = config$intermediate_size,
    n_head = config$num_attention_heads,
    n_kv_head = config$num_key_value_heads,
    head_dim = config$head_dim %||% (config$hidden_size %/% config$num_attention_heads),
    n_layer = config$num_hidden_layers,
    max_pos = config$max_position_embeddings,
    rmsnorm_eps = config$rms_norm_eps %||% 1e-5,
    rope_base = rope_params$rope_theta %||% 1e9,
    rope_factor = rope_params$factor %||% 16,
    rope_beta_fast = rope_params$beta_fast %||% 32,
    rope_beta_slow = rope_params$beta_slow %||% 1,
    rope_original_max_pos = rope_params$original_max_position_embeddings %||% 16384,
    rope_mscale = rope_params$mscale %||% 1,
    rope_mscale_all_dim = rope_params$mscale_all_dim %||% 1
  )
}

#' @describeIn ministral Initializes and loads pretrained weights from HF Hub
#' @export
ministral_from_pretrained <- function(identifier, revision = "main") {
  with_device(device = "meta", {
    model <- ministral_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- ministral_hf_weights_remap(state_dict)
  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

ministral_hf_weights_remap <- function(state_dict) {
  nms <- names(state_dict)

  # Handle multimodal models (language_model prefix)
  nms <- gsub("^language_model\\.", "", nms)

  # Standard remapping
  nms <- gsub("model.embed_tokens.weight", "transformer.wte.weight", nms, fixed = TRUE)
  nms <- gsub("model.layers", "transformer.h", nms, fixed = TRUE)
  nms <- gsub("self_attn", "attn", nms, fixed = TRUE)
  nms <- gsub("input_layernorm", "ln_1", nms, fixed = TRUE)
  nms <- gsub("post_attention_layernorm", "ln_2", nms, fixed = TRUE)
  nms <- gsub("model.norm", "transformer.ln_f", nms, fixed = TRUE)

  names(state_dict) <- nms

  # Filter out non-language model weights
  keep <- !grepl("vision_tower|multi_modal_projector|scale", names(state_dict))
  state_dict[keep]
}
