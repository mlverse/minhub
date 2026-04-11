# References:
# - https://huggingface.co/Qwen/Qwen3.5-0.8B
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py

#' @noRd
#' @importFrom zeallot %<-%
#' @importFrom purrr map
#' @import torch
NULL

# Qwen3Next-style RMSNorm with unit offset (weight initialized to zeros, output = (1 + w) * norm(x))
nn_qwen3_rmsnorm <- nn_module(
  initialize = function(hidden_size, eps = 1e-6) {
    self$weight <- nn_parameter(torch_zeros(hidden_size))
    self$eps <- eps
  },
  forward = function(x) {
    dtype <- x$dtype
    variance <- x$to(dtype = "float32")$pow(2)$mean(-1, keepdim = TRUE)
    x <- x * torch_rsqrt(variance + self$eps)
    ((1 + self$weight$float()) * x)$to(dtype = dtype)
  }
)

# Gated RMSNorm: normalizes hidden_states then multiplies by silu(gate)
nn_qwen3_rmsnorm_gated <- nn_module(
  initialize = function(hidden_size, eps = 1e-6) {
    self$weight <- nn_parameter(torch_ones(hidden_size))
    self$eps <- eps
  },
  forward = function(hidden_states, gate) {
    dtype <- hidden_states$dtype
    hidden_states <- hidden_states$to(dtype = "float32")
    variance <- hidden_states$pow(2)$mean(-1, keepdim = TRUE)
    hidden_states <- hidden_states * torch_rsqrt(variance + self$eps)
    hidden_states <- (self$weight * hidden_states)$to(dtype = dtype)
    hidden_states <- hidden_states * nnf_silu(gate$to(dtype = "float32"))
    hidden_states$to(dtype = dtype)
  }
)

qwen3_rotate_half <- function(x) {
  c(x1, x2) %<-% torch_split(x, x$size(-1) / 2, -1)
  torch_cat(list(-x2, x1), dim = -1)
}

# RoPE with partial rotary factor: only rotates the first `rotary_dim` dims
# Computes cos/sin on the fly to avoid large pre-allocated buffers
nn_qwen3_rotary_embedding <- nn_module(
  initialize = function(head_dim, max_pos, base = 10000, partial_rotary_factor = 1.0) {
    self$head_dim <- head_dim
    self$max_pos <- max_pos
    self$base <- base
    self$rotary_dim <- as.integer(head_dim * partial_rotary_factor)

    self$compute_inv_freq()
  },
  compute_inv_freq = function() {
    dim <- self$rotary_dim
    inv_freq <- 1 / (self$base ^ (torch_arange(0, dim - 1, step = 2) / dim))
    self$inv_freq <- nn_buffer(inv_freq, persistent = FALSE)
  },
  .load_from_state_dict = function(...) {
    super$.load_from_state_dict(...)
    self$compute_inv_freq()
  },
  forward = function(x) {
    c(b, nh, t, d) %<-% x$shape
    dim <- self$rotary_dim

    freqs <- torch_arange(start = 0, end = t - 1, device = x$device)$
      float()$outer(self$inv_freq)$
      view(c(1, 1, t, dim / 2))
    emb <- torch_cat(list(freqs, freqs), dim = -1)
    cos <- emb$cos()
    sin <- emb$sin()

    if (self$rotary_dim < self$head_dim) {
      x_rot <- x[,,,1:self$rotary_dim]
      x_pass <- x[,,,(self$rotary_dim + 1):d]
      x_rot <- x_rot * cos + qwen3_rotate_half(x_rot) * sin
      torch_cat(list(x_rot, x_pass), dim = -1)
    } else {
      x * cos + qwen3_rotate_half(x) * sin
    }
  }
)

qwen3_repeat_kv <- function(hidden_states, n_rep) {
  if (n_rep == 1) return(hidden_states)
  c(batch, num_kv_heads, seq_len, head_dim) %<-% hidden_states$shape
  hidden_states$unsqueeze(3)$
    expand(c(batch, num_kv_heads, n_rep, seq_len, head_dim))$
    reshape(c(batch, num_kv_heads * n_rep, seq_len, head_dim))
}

# Full attention with output gate (Qwen3Next-style)
# q_proj outputs 2x: half for query, half for sigmoid gate on output
nn_qwen3_attention <- nn_module(
  initialize = function(n_embd, n_head, n_kv_head, head_dim, max_pos,
                        rope_base, partial_rotary_factor, rmsnorm_eps) {
    self$n_head <- n_head
    self$n_kv_head <- n_kv_head
    self$head_dim <- head_dim
    self$n_kv_groups <- n_head %/% n_kv_head

    self$rotary <- nn_qwen3_rotary_embedding(head_dim, max_pos, rope_base, partial_rotary_factor)

    # q_proj outputs 2x head_dim: first half = query, second half = gate
    self$q_proj <- nn_linear(n_embd, n_head * head_dim * 2, bias = FALSE)
    self$k_proj <- nn_linear(n_embd, n_kv_head * head_dim, bias = FALSE)
    self$v_proj <- nn_linear(n_embd, n_kv_head * head_dim, bias = FALSE)
    self$o_proj <- nn_linear(n_head * head_dim, n_embd, bias = FALSE)

    self$q_norm <- nn_qwen3_rmsnorm(head_dim, eps = rmsnorm_eps)
    self$k_norm <- nn_qwen3_rmsnorm(head_dim, eps = rmsnorm_eps)
  },
  forward = function(x) {
    c(b, t, h) %<-% x$shape

    # q_proj -> split into query and gate
    qg <- self$q_proj(x)$view(c(b, t, self$n_head, self$head_dim * 2))
    c(q, gate) %<-% torch_chunk(qg, 2, dim = -1) # each (b, t, n_head, head_dim)
    gate <- gate$reshape(c(b, t, -1)) # (b, t, n_head * head_dim)

    q <- self$q_norm(q)$transpose(2, 3) # (b, n_head, t, head_dim)
    k <- self$k_proj(x)$view(c(b, t, self$n_kv_head, self$head_dim))
    k <- self$k_norm(k)$transpose(2, 3)
    v <- self$v_proj(x)$view(c(b, t, self$n_kv_head, self$head_dim))$transpose(2, 3)

    q <- self$rotary(q)$to(dtype = "float")
    k <- self$rotary(k)$to(dtype = "float")

    k <- qwen3_repeat_kv(k, self$n_kv_groups)
    v <- qwen3_repeat_kv(v, self$n_kv_groups)

    att <- torch_matmul(q, k$transpose(-2, -1)) * (1 / sqrt(self$head_dim))
    # Generate causal mask on the fly (max_pos is too large to pre-allocate)
    causal_mask <- torch_ones(t, t, dtype = torch_bool(), device = x$device)$triu(diagonal = 1)
    att <- att$masked_fill(causal_mask$view(c(1, 1, t, t)), -Inf)
    att <- nnf_softmax(att, dim = -1)$to(dtype = v$dtype)

    y <- torch_matmul(att, v)$transpose(2, 3)$contiguous()$
      reshape(c(b, t, -1))

    # Apply sigmoid gate
    y <- y * torch_sigmoid(gate)
    self$o_proj(y)
  }
)

# Recurrent gated delta rule (simple loop-based implementation)
qwen3_recurrent_gated_delta_rule <- function(query, key, value, g, beta) {
  # Inputs: (batch, seq, heads, dim) for q/k/v; (batch, seq, heads) for g/beta
  # L2 normalize q and k
  query <- nnf_normalize(query, p = 2, dim = -1, eps = 1e-6)
  key <- nnf_normalize(key, p = 2, dim = -1, eps = 1e-6)

  # Transpose to (batch, heads, seq, dim)
  query <- query$transpose(2, 3)$contiguous()$to(dtype = "float32")
  key <- key$transpose(2, 3)$contiguous()$to(dtype = "float32")
  value <- value$transpose(2, 3)$contiguous()$to(dtype = "float32")
  beta <- beta$transpose(2, 3)$contiguous()$to(dtype = "float32")
  g <- g$transpose(2, 3)$contiguous()$to(dtype = "float32")

  c(batch_size, num_heads, seq_len, k_head_dim) %<-% key$shape
  v_head_dim <- value$shape[4]

  scale <- 1 / sqrt(k_head_dim)
  query <- query * scale

  state <- torch_zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                        device = query$device, dtype = query$dtype)
  outputs <- vector("list", seq_len)

  for (i in seq_len(seq_len)) {
    q_t <- query[,,i,]           # (batch, heads, k_dim)
    k_t <- key[,,i,]             # (batch, heads, k_dim)
    v_t <- value[,,i,]           # (batch, heads, v_dim)
    g_t <- g[,,i]$exp()$unsqueeze(-1)$unsqueeze(-1) # (batch, heads, 1, 1)
    beta_t <- beta[,,i]$unsqueeze(-1)                # (batch, heads, 1)

    state <- state * g_t
    # kv_mem = (state * k_t[:,:,:,None]).sum(dim=-2) => k_t @ state
    kv_mem <- torch_einsum("bhkv,bhk->bhv", list(state, k_t))
    delta <- (v_t - kv_mem) * beta_t
    # state += k_t[:,:,:,None] * delta[:,:,None,:]
    state <- state + torch_einsum("bhk,bhv->bhkv", list(k_t, delta))
    # output = (state * q_t[:,:,:,None]).sum(dim=-2) => q_t @ state
    outputs[[i]] <- torch_einsum("bhkv,bhk->bhv", list(state, q_t))
  }

  # Stack: list of (batch, heads, v_dim) -> (batch, heads, seq, v_dim)
  out <- torch_stack(outputs, dim = 3)
  # Back to (batch, seq, heads, v_dim)
  out$transpose(2, 3)$contiguous()
}

# Gated Delta Net linear attention
nn_qwen3_gated_delta_net <- nn_module(
  initialize = function(n_embd, n_k_heads, n_v_heads, k_head_dim, v_head_dim,
                        conv_kernel_size, rmsnorm_eps, hidden_act = "silu") {
    self$n_embd <- n_embd
    self$n_k_heads <- n_k_heads
    self$n_v_heads <- n_v_heads
    self$k_head_dim <- k_head_dim
    self$v_head_dim <- v_head_dim
    self$key_dim <- n_k_heads * k_head_dim
    self$value_dim <- n_v_heads * v_head_dim

    conv_dim <- self$key_dim * 2 + self$value_dim
    self$conv1d <- nn_conv1d(
      in_channels = conv_dim, out_channels = conv_dim,
      kernel_size = conv_kernel_size, groups = conv_dim,
      bias = FALSE, padding = conv_kernel_size - 1
    )

    self$dt_bias <- nn_parameter(torch_ones(n_v_heads))
    self$A_log <- nn_parameter(torch_log(torch_empty(n_v_heads)$uniform_(0, 16)))

    self$norm <- nn_qwen3_rmsnorm_gated(v_head_dim, eps = rmsnorm_eps)
    self$out_proj <- nn_linear(self$value_dim, n_embd, bias = FALSE)

    self$in_proj_qkv <- nn_linear(n_embd, self$key_dim * 2 + self$value_dim, bias = FALSE)
    self$in_proj_z <- nn_linear(n_embd, self$value_dim, bias = FALSE)
    self$in_proj_b <- nn_linear(n_embd, n_v_heads, bias = FALSE)
    self$in_proj_a <- nn_linear(n_embd, n_v_heads, bias = FALSE)
  },
  forward = function(x) {
    c(batch_size, seq_len, .) %<-% x$shape

    mixed_qkv <- self$in_proj_qkv(x)$transpose(2, 3) # (batch, channels, seq)
    z <- self$in_proj_z(x)$reshape(c(batch_size, seq_len, -1, self$v_head_dim))
    b <- self$in_proj_b(x)
    a <- self$in_proj_a(x)

    # Causal conv1d + SiLU
    mixed_qkv <- nnf_silu(self$conv1d(mixed_qkv)[,,1:seq_len])
    mixed_qkv <- mixed_qkv$transpose(2, 3) # back to (batch, seq, channels)

    c(query, key, value) %<-% torch_split(
      mixed_qkv, c(self$key_dim, self$key_dim, self$value_dim), dim = -1
    )

    query <- query$reshape(c(batch_size, seq_len, -1, self$k_head_dim))
    key <- key$reshape(c(batch_size, seq_len, -1, self$k_head_dim))
    value <- value$reshape(c(batch_size, seq_len, -1, self$v_head_dim))

    beta <- b$sigmoid()
    g <- -self$A_log$float()$exp() * nnf_softplus(a$float() + self$dt_bias)

    # Repeat interleave k heads to match v heads
    if (self$n_v_heads %/% self$n_k_heads > 1) {
      n_rep <- self$n_v_heads %/% self$n_k_heads
      query <- query$repeat_interleave(n_rep, dim = 3)
      key <- key$repeat_interleave(n_rep, dim = 3)
    }

    initial_dtype <- query$dtype
    core_out <- qwen3_recurrent_gated_delta_rule(query, key, value, g, beta)

    # Gated RMSNorm: reshape to 2D, apply, reshape back
    core_out_2d <- core_out$reshape(c(-1, self$v_head_dim))
    z_2d <- z$reshape(c(-1, self$v_head_dim))
    core_out_2d <- self$norm(core_out_2d, z_2d)
    core_out <- core_out_2d$reshape(c(batch_size, seq_len, -1))$to(dtype = initial_dtype)

    self$out_proj(core_out)
  }
)

nn_qwen3_mlp <- nn_module(
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

nn_qwen3_layer <- nn_module(
  initialize = function(layer_type, n_embd, n_inter, n_head, n_kv_head, head_dim,
                        max_pos, rmsnorm_eps, rope_base, partial_rotary_factor,
                        n_k_heads, n_v_heads, k_head_dim, v_head_dim,
                        conv_kernel_size) {
    self$layer_type <- layer_type
    self$ln_1 <- nn_qwen3_rmsnorm(n_embd, rmsnorm_eps)
    self$ln_2 <- nn_qwen3_rmsnorm(n_embd, rmsnorm_eps)

    if (layer_type == "full_attention") {
      self$attn <- nn_qwen3_attention(
        n_embd, n_head, n_kv_head, head_dim, max_pos,
        rope_base, partial_rotary_factor, rmsnorm_eps
      )
    } else {
      self$linear_attn <- nn_qwen3_gated_delta_net(
        n_embd, n_k_heads, n_v_heads, k_head_dim, v_head_dim,
        conv_kernel_size, rmsnorm_eps
      )
    }

    self$mlp <- nn_qwen3_mlp(n_embd, n_inter)
  },
  forward = function(x) {
    if (self$layer_type == "full_attention") {
      x <- x + self$attn(self$ln_1(x))
    } else {
      x <- x + self$linear_attn(self$ln_1(x))
    }
    x + self$mlp(self$ln_2(x))
  }
)

nn_qwen3_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_inter, n_head, n_kv_head, head_dim,
                        n_layer, max_pos, rmsnorm_eps, rope_base,
                        partial_rotary_factor, layer_types,
                        n_k_heads, n_v_heads, k_head_dim, v_head_dim,
                        conv_kernel_size) {
    self$transformer <- nn_module_dict(list(
      wte = nn_embedding(vocab_size, n_embd),
      h = nn_sequential(!!!map(
        seq_len(n_layer),
        \(i) nn_qwen3_layer(
          layer_type = layer_types[i],
          n_embd = n_embd, n_inter = n_inter,
          n_head = n_head, n_kv_head = n_kv_head, head_dim = head_dim,
          max_pos = max_pos, rmsnorm_eps = rmsnorm_eps,
          rope_base = rope_base, partial_rotary_factor = partial_rotary_factor,
          n_k_heads = n_k_heads, n_v_heads = n_v_heads,
          k_head_dim = k_head_dim, v_head_dim = v_head_dim,
          conv_kernel_size = conv_kernel_size
        )
      )),
      ln_f = nn_qwen3_rmsnorm(n_embd, rmsnorm_eps)
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

#' Qwen3.5
#'
#' Initializes a Qwen3.5 model with hybrid linear/full attention (Gated Delta Net)
#'
#' @param vocab_size Vocabulary size.
#' @param n_embd Embedding dimension.
#' @param n_inter Intermediate size in MLP.
#' @param n_head Number of attention heads (full attention).
#' @param n_kv_head Number of key/value heads (full attention GQA).
#' @param head_dim Dimension of each attention head (full attention).
#' @param n_layer Number of transformer layers.
#' @param max_pos Maximum position embeddings.
#' @param rmsnorm_eps Epsilon for RMSNorm.
#' @param rope_base Base for rotary embeddings.
#' @param partial_rotary_factor Fraction of head_dim to apply rotary embeddings to.
#' @param layer_types Character vector of layer types ("linear_attention" or "full_attention").
#' @param n_k_heads Number of key heads (linear attention).
#' @param n_v_heads Number of value heads (linear attention).
#' @param k_head_dim Key head dimension (linear attention).
#' @param v_head_dim Value head dimension (linear attention).
#' @param conv_kernel_size Causal conv1d kernel size (linear attention).
#' @param identifier HuggingFace model identifier.
#' @param revision HuggingFace model revision.
#' @returns An initialized [torch::nn_module()].
#' @export
qwen3 <- function(vocab_size = 248320, n_embd = 1024, n_inter = 3584,
                   n_head = 8, n_kv_head = 2, head_dim = 256,
                   n_layer = 24, max_pos = 262144, rmsnorm_eps = 1e-6,
                   rope_base = 1e7, partial_rotary_factor = 0.25,
                   layer_types = rep(c(rep("linear_attention", 3), "full_attention"), 6),
                   n_k_heads = 16, n_v_heads = 16,
                   k_head_dim = 128, v_head_dim = 128,
                   conv_kernel_size = 4) {
  nn_qwen3_model(
    vocab_size, n_embd, n_inter, n_head, n_kv_head, head_dim,
    n_layer, max_pos, rmsnorm_eps, rope_base, partial_rotary_factor,
    layer_types, n_k_heads, n_v_heads, k_head_dim, v_head_dim, conv_kernel_size
  )
}

#' @describeIn qwen3 Initializes from HuggingFace config
#' @export
qwen3_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  # Handle multimodal config (text_config nested)
  if (!is.null(config$text_config)) {
    text_config <- config$text_config
  } else {
    text_config <- config
  }

  if (!text_config$model_type %in% c("qwen3_5_text", "qwen3_5"))
    cli::cli_abort("Unsupported model_type: {.val {text_config$model_type}}")

  if (text_config$hidden_act != "silu")
    cli::cli_abort("Unsupported hidden_act: {.val {text_config$hidden_act}}")

  rope_params <- text_config$rope_parameters %||% text_config

  # Build layer_types from full_attention_interval
  full_attn_interval <- text_config$full_attention_interval %||% 4
  n_layer <- text_config$num_hidden_layers
  layer_types <- ifelse(
    seq_len(n_layer) %% full_attn_interval == 0,
    "full_attention",
    "linear_attention"
  )

  # Linear attention config
  linear_config <- text_config$linear_attention_config %||% list()

  qwen3(
    vocab_size = text_config$vocab_size,
    n_embd = text_config$hidden_size,
    n_inter = text_config$intermediate_size,
    n_head = text_config$num_attention_heads,
    n_kv_head = text_config$num_key_value_heads,
    head_dim = text_config$head_dim %||% (text_config$hidden_size %/% text_config$num_attention_heads),
    n_layer = n_layer,
    max_pos = text_config$max_position_embeddings,
    rmsnorm_eps = text_config$rms_norm_eps %||% 1e-6,
    rope_base = rope_params$rope_theta %||% 1e7,
    partial_rotary_factor = rope_params$partial_rotary_factor %||% 0.25,
    layer_types = layer_types,
    n_k_heads = linear_config$num_key_heads %||% 16,
    n_v_heads = linear_config$num_value_heads %||% 16,
    k_head_dim = linear_config$key_head_dim %||% 128,
    v_head_dim = linear_config$value_head_dim %||% 128,
    conv_kernel_size = linear_config$conv_kernel_dim %||% 4
  )
}

#' @describeIn qwen3 Initializes and loads pretrained weights from HF Hub
#' @export
qwen3_from_pretrained <- function(identifier, revision = "main") {
  with_device(device = "meta", {
    model <- qwen3_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- qwen3_hf_weights_remap(state_dict)
  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

qwen3_hf_weights_remap <- function(state_dict) {
  nms <- names(state_dict)

  # Filter out non-text weights (vision, projector, mtp, etc.)
  keep <- !grepl("visual|vision|multi_modal|mtp\\.", nms)
  state_dict <- state_dict[keep]
  nms <- names(state_dict)

  # Strip multimodal prefix: model.language_model. -> model.
  nms <- gsub("^model\\.language_model\\.", "model.", nms)

  # Standard remapping
  nms <- gsub("model.embed_tokens.weight", "transformer.wte.weight", nms, fixed = TRUE)
  nms <- gsub("model.layers", "transformer.h", nms, fixed = TRUE)
  nms <- gsub("self_attn", "attn", nms, fixed = TRUE)
  nms <- gsub("input_layernorm", "ln_1", nms, fixed = TRUE)
  nms <- gsub("post_attention_layernorm", "ln_2", nms, fixed = TRUE)
  nms <- gsub("model.norm", "transformer.ln_f", nms, fixed = TRUE)

  names(state_dict) <- nms

  # Tie lm_head to embedding if not present
  if (!"lm_head.weight" %in% names(state_dict)) {
    state_dict["lm_head.weight"] <- state_dict["transformer.wte.weight"]
  }

  state_dict
}
