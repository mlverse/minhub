# References:
# - https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
# - https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py

#' @noRd
#' @importFrom purrr map
#' @import torch
NULL

gptoss_default_config <- function() {
  list(
    num_hidden_layers = 36L,
    num_experts = 128L,
    experts_per_token = 4L,
    vocab_size = 201088L,
    hidden_size = 2880L,
    intermediate_size = 2880L,
    swiglu_limit = 7.0,
    head_dim = 64L,
    num_attention_heads = 64L,
    num_key_value_heads = 8L,
    sliding_window = 128L,
    initial_context_length = 4096L,
    rope_theta = 150000.0,
    rope_scaling_factor = 32.0,
    rope_ntk_alpha = 1.0,
    rope_ntk_beta = 32.0
  )
}

gptoss_normalize_config <- function(config) {
  defaults <- gptoss_default_config()

  # Map HuggingFace config names to internal names
  hf_to_internal <- c(
    num_local_experts = "num_experts",
    num_experts_per_tok = "experts_per_token"
  )

  for (hf_nm in names(hf_to_internal)) {
    if (hf_nm %in% names(config)) {
      internal_nm <- hf_to_internal[[hf_nm]]
      defaults[[internal_nm]] <- config[[hf_nm]]
    }
  }

  # Handle nested rope_scaling config from HF
  if ("rope_scaling" %in% names(config) && is.list(config$rope_scaling)) {
    rs <- config$rope_scaling
    if (!is.null(rs$factor)) defaults$rope_scaling_factor <- rs$factor
    if (!is.null(rs$beta_slow)) defaults$rope_ntk_alpha <- rs$beta_slow
    if (!is.null(rs$beta_fast)) defaults$rope_ntk_beta <- rs$beta_fast
  }

  # Copy direct matches
  keys <- intersect(names(defaults), names(config))
  for (nm in keys) {
    defaults[[nm]] <- config[[nm]]
  }

  defaults
}

nn_gptoss_rmsnorm <- nn_module(
  initialize = function(num_features, eps = 1e-5) {
    self$num_features <- num_features
    self$eps <- eps
    self$scale <- nn_parameter(torch_ones(num_features, dtype = torch_float()))
  },
  forward = function(x) {
    dtype <- x$dtype
    t <- x$to(dtype = torch_float())
    t <- t * torch_rsqrt(t$pow(2)$mean(dim = -1, keepdim = TRUE) + self$eps)
    (t * self$scale)$to(dtype = dtype)
  }
)

gptoss_apply_rotary_emb <- function(x, cos, sin) {
  cos <- cos$unsqueeze(-2)$to(dtype = x$dtype)
  sin <- sin$unsqueeze(-2)$to(dtype = x$dtype)
  parts <- torch_chunk(x, chunks = 2, dim = -1)
  x1 <- parts[[1]]
  x2 <- parts[[2]]
  o1 <- x1 * cos - x2 * sin
  o2 <- x2 * cos + x1 * sin
  torch_cat(list(o1, o2), dim = -1)
}

nn_gptoss_rotary_embedding <- nn_module(
  initialize = function(
    head_dim,
    base,
    initial_context_length = 4096L,
    scaling_factor = 1.0,
    ntk_alpha = 1.0,
    ntk_beta = 32.0
  ) {
    self$head_dim <- head_dim
    self$base <- base
    self$initial_context_length <- initial_context_length
    self$scaling_factor <- scaling_factor
    self$ntk_alpha <- ntk_alpha
    self$ntk_beta <- ntk_beta
  },
  compute_concentration_and_inv_freq = function(device) {
    freq <- self$base ^ (
      torch_arange(
        0,
        self$head_dim - 1L,
        step = 2L,
        device = device,
        dtype = torch_float()
      ) / self$head_dim
    )

    if (self$scaling_factor > 1.0) {
      concentration <- 0.1 * log(self$scaling_factor) + 1.0

      d_half <- self$head_dim / 2
      low <- d_half *
        log(self$initial_context_length / (self$ntk_beta * 2 * pi)) /
        log(self$base)
      high <- d_half *
        log(self$initial_context_length / (self$ntk_alpha * 2 * pi)) /
        log(self$base)

      interpolation <- 1.0 / (self$scaling_factor * freq)
      extrapolation <- 1.0 / freq

      ramp <- (torch_arange(
        start = 0,
        end = d_half - 1L,
        device = device,
        dtype = torch_float()
      ) - low) /
        (high - low)
      mask <- 1 - ramp$clamp(min = 0, max = 1)
      inv_freq <- interpolation * (1 - mask) + extrapolation * mask
    } else {
      concentration <- 1.0
      inv_freq <- 1.0 / freq
    }

    list(concentration = concentration, inv_freq = inv_freq)
  },
  compute_cos_sin = function(num_tokens, device) {
    values <- self$compute_concentration_and_inv_freq(device)
    concentration <- values$concentration
    inv_freq <- values$inv_freq

    t <- torch_arange(
      start = 0,
      end = num_tokens - 1L,
      device = device,
      dtype = torch_float()
    )
    freqs <- torch_einsum("i,j->ij", list(t, inv_freq))
    cos <- freqs$cos() * concentration
    sin <- freqs$sin() * concentration
    list(cos = cos, sin = sin)
  },
  forward = function(query, key) {
    num_tokens <- query$size(1)
    values <- self$compute_cos_sin(num_tokens, query$device)
    cos <- values$cos
    sin <- values$sin

    query_shape <- query$shape
    query <- query$view(c(num_tokens, -1L, self$head_dim))
    query <- gptoss_apply_rotary_emb(query, cos, sin)
    query <- query$reshape(query_shape)

    key_shape <- key$shape
    key <- key$view(c(num_tokens, -1L, self$head_dim))
    key <- gptoss_apply_rotary_emb(key, cos, sin)
    key <- key$reshape(key_shape)

    list(query = query, key = key)
  }
)

gptoss_sdpa <- function(Q, K, V, S, sm_scale, sliding_window = 0L) {
  n_tokens <- Q$size(1)
  n_heads <- Q$size(2)
  q_mult <- Q$size(3)
  d_head <- Q$size(4)

  K <- K$unsqueeze(3)$expand(c(-1L, -1L, q_mult, -1L))
  V <- V$unsqueeze(3)$expand(c(-1L, -1L, q_mult, -1L))
  S <- S$reshape(c(n_heads, q_mult, 1L, 1L))$expand(c(-1L, -1L, n_tokens, -1L))

  mask <- torch_triu(
    torch_full(c(n_tokens, n_tokens), -Inf, device = Q$device, dtype = Q$dtype),
    diagonal = 1L
  )
  if (sliding_window > 0) {
    mask <- mask + torch_tril(
      torch_full(c(n_tokens, n_tokens), -Inf, device = Q$device, dtype = Q$dtype),
      diagonal = -sliding_window
    )
  }

  qk <- torch_einsum("qhmd,khmd->hmqk", list(Q, K))
  qk <- qk * sm_scale
  qk <- qk + mask$unsqueeze(1)$unsqueeze(1)
  qk <- torch_cat(list(qk, S), dim = -1)
  w <- nnf_softmax(qk, dim = -1)
  w <- w[,,,1:n_tokens]
  attn <- torch_einsum("hmqk,khmd->qhmd", list(w, V))
  attn$reshape(c(n_tokens, -1L))
}

nn_gptoss_attention_block <- nn_module(
  initialize = function(config, layer_idx = 0L) {
    self$head_dim <- config$head_dim
    self$num_attention_heads <- config$num_attention_heads
    self$num_key_value_heads <- config$num_key_value_heads
    self$sliding_window <- if (layer_idx %% 2L == 0L) config$sliding_window else 0L
    self$sm_scale <- 1 / sqrt(config$head_dim)

    self$sinks <- nn_parameter(
      torch_empty(config$num_attention_heads, dtype = torch_bfloat16())
    )
    self$norm <- nn_gptoss_rmsnorm(config$hidden_size)

    qkv_dim <- config$head_dim * (
      config$num_attention_heads + 2L * config$num_key_value_heads
    )
    self$qkv <- nn_linear(config$hidden_size, qkv_dim)
    self$out <- nn_linear(config$head_dim * config$num_attention_heads, config$hidden_size)
    self$qkv$to(dtype = torch_bfloat16())
    self$out$to(dtype = torch_bfloat16())

    self$rope <- nn_gptoss_rotary_embedding(
      config$head_dim,
      config$rope_theta,
      initial_context_length = config$initial_context_length,
      scaling_factor = config$rope_scaling_factor,
      ntk_alpha = config$rope_ntk_alpha,
      ntk_beta = config$rope_ntk_beta
    )
  },
  forward = function(x) {
    t <- self$norm(x)
    qkv <- self$qkv(t)

    q <- qkv[, 1:(self$num_attention_heads * self$head_dim)]$contiguous()
    k <- qkv[,
      (self$num_attention_heads * self$head_dim + 1):(
        (self$num_attention_heads + self$num_key_value_heads) * self$head_dim
      )
    ]$contiguous()
    v <- qkv[,
      ((self$num_attention_heads + self$num_key_value_heads) * self$head_dim + 1):(
        (self$num_attention_heads + 2L * self$num_key_value_heads) * self$head_dim
      )
    ]$contiguous()

    q <- q$view(c(
      -1L,
      self$num_key_value_heads,
      self$num_attention_heads / self$num_key_value_heads,
      self$head_dim
    ))
    k <- k$view(c(-1L, self$num_key_value_heads, self$head_dim))
    v <- v$view(c(-1L, self$num_key_value_heads, self$head_dim))

    rotary <- self$rope(q, k)
    q <- rotary$query
    k <- rotary$key

    t <- gptoss_sdpa(q, k, v, self$sinks, self$sm_scale, self$sliding_window)
    t <- self$out(t)
    x + t
  }
)

gptoss_swiglu <- function(x, alpha = 1.702, limit = 7.0) {
  last_dim <- x$size(length(x$shape))
  x_glu <- x[,,seq(1L, last_dim, by = 2L)]
  x_linear <- x[,,seq(2L, last_dim, by = 2L)]
  x_glu <- x_glu$clamp(max = limit)
  x_linear <- x_linear$clamp(min = -limit, max = limit)
  out_glu <- x_glu * torch_sigmoid(x_glu * alpha)
  out_glu * (x_linear + 1)
}

nn_gptoss_mlp_block <- nn_module(
  initialize = function(config) {
    self$num_experts <- config$num_experts
    self$experts_per_token <- config$experts_per_token
    self$swiglu_limit <- config$swiglu_limit
    self$world_size <- 1L

    if ((config$intermediate_size %% self$world_size) != 0L) {
      cli::cli_abort("{.arg intermediate_size} must be divisible by {.arg world_size}.")
    }

    self$norm <- nn_gptoss_rmsnorm(config$hidden_size)
    self$gate <- nn_linear(config$hidden_size, config$num_experts)
    self$gate$to(dtype = torch_bfloat16())

    self$mlp1_weight <- nn_parameter(
      torch_empty(
        c(
          config$num_experts,
          config$intermediate_size * 2L / self$world_size,
          config$hidden_size
        ),
        dtype = torch_bfloat16()
      )
    )
    self$mlp1_bias <- nn_parameter(
      torch_empty(
        c(config$num_experts, config$intermediate_size * 2L / self$world_size),
        dtype = torch_bfloat16()
      )
    )
    self$mlp2_weight <- nn_parameter(
      torch_empty(
        c(
          config$num_experts,
          config$hidden_size,
          config$intermediate_size / self$world_size
        ),
        dtype = torch_bfloat16()
      )
    )
    self$mlp2_bias <- nn_parameter(
      torch_empty(
        c(config$num_experts, config$hidden_size),
        dtype = torch_bfloat16()
      )
    )
  },
  forward = function(x) {
    t <- self$norm(x)
    g <- self$gate(t)

    experts <- torch_topk(g, k = self$experts_per_token, dim = -1, sorted = TRUE)
    expert_weights <- nnf_softmax(experts[[1]], dim = -1)
    expert_indices <- experts[[2]]

    mlp1_weight <- self$mlp1_weight[expert_indices, ]
    mlp1_bias <- self$mlp1_bias[expert_indices, ]
    t <- torch_einsum("beck,bk->bec", list(mlp1_weight, t)) + mlp1_bias
    t <- gptoss_swiglu(t, limit = self$swiglu_limit)

    mlp2_weight <- self$mlp2_weight[expert_indices, ]
    mlp2_bias <- self$mlp2_bias[expert_indices, ]
    t <- torch_einsum("beck,bek->bec", list(mlp2_weight, t))
    t <- t + mlp2_bias

    t <- torch_einsum("bec,be->bc", list(t, expert_weights))
    x + t
  }
)

nn_gptoss_transformer_block <- nn_module(
  initialize = function(config, layer_idx) {
    self$attn <- nn_gptoss_attention_block(config, layer_idx = layer_idx)
    self$mlp <- nn_gptoss_mlp_block(config)
  },
  forward = function(x) {
    x <- self$attn(x)
    x <- self$mlp(x)
    x
  }
)

nn_gptoss_model <- nn_module(
  initialize = function(config) {
    self$embedding <- nn_embedding(config$vocab_size, config$hidden_size)
    self$embedding$to(dtype = torch_bfloat16())

    self$block <- nn_sequential(!!!map(
      0:(config$num_hidden_layers - 1L),
      \(idx) nn_gptoss_transformer_block(config, idx)
    ))

    self$norm <- nn_gptoss_rmsnorm(config$hidden_size)
    self$unembedding <- nn_linear(config$hidden_size, config$vocab_size, bias = FALSE)
    self$unembedding$to(dtype = torch_bfloat16())
  },
  forward = function(x) {
    batched <- FALSE
    if (length(x$shape) == 2L) {
      if (x$size(1) != 1L) {
        cli::cli_abort(c(
          "{.fn gptoss} currently supports either a token vector or a single-row token matrix.",
          i = "Got {.arg x} with shape {.val {paste(x$shape, collapse = ' x ')}}."
        ))
      }
      batched <- TRUE
      x <- x$squeeze(1)
    }

    x <- self$embedding(x)
    x <- self$block(x)
    x <- self$norm(x)
    x <- self$unembedding(x)

    if (batched) {
      x <- x$unsqueeze(1)
    }
    x
  }
)

gptoss_fp4_values <- c(
  +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
)


gptoss_hf_weights_remap <- function(state_dict) {
  # First, decode MXFP4 weights and remove blocks/scales entries
  # HF uses underscore suffix: _blocks, _scales
  param_names <- unique(gsub("_(blocks|scales)$", "", names(state_dict)))
  decoded_dict <- purrr::map(param_names, function(nm) {
    blocks_name <- paste0(nm, "_blocks")
    scales_name <- paste0(nm, "_scales")

    if (blocks_name %in% names(state_dict) && scales_name %in% names(state_dict)) {
      gptoss_decode_mxfp4(state_dict[[blocks_name]], state_dict[[scales_name]])
    } else {
      state_dict[[nm]]
    }
  }) |> setNames(param_names)

  # Build the remapped state dict with model-expected names
  result <- list()

  # Embedding and output layers
  result[["embedding.weight"]] <- decoded_dict[["model.embed_tokens.weight"]]
  result[["norm.scale"]] <- decoded_dict[["model.norm.weight"]]
  result[["unembedding.weight"]] <- decoded_dict[["lm_head.weight"]]

  # Find all layer indices
  layer_nums <- gsub("model\\.layers\\.([0-9]+)\\..*", "\\1", names(decoded_dict))
  layer_nums <- layer_nums[grepl("^[0-9]+$", layer_nums)]
  layer_nums <- sort(unique(as.integer(layer_nums)))

  for (layer_idx in layer_nums) {
    prefix <- paste0("model.layers.", layer_idx, ".")
    block_prefix <- paste0("block.", layer_idx, ".")

    # Attention norm
    result[[paste0(block_prefix, "attn.norm.scale")]] <-
      decoded_dict[[paste0(prefix, "input_layernorm.weight")]]

    # Attention sinks
    result[[paste0(block_prefix, "attn.sinks")]] <-
      decoded_dict[[paste0(prefix, "self_attn.sinks")]]

    # Concatenate q, k, v projections into qkv
    q_weight <- decoded_dict[[paste0(prefix, "self_attn.q_proj.weight")]]
    k_weight <- decoded_dict[[paste0(prefix, "self_attn.k_proj.weight")]]
    v_weight <- decoded_dict[[paste0(prefix, "self_attn.v_proj.weight")]]
    result[[paste0(block_prefix, "attn.qkv.weight")]] <-
      torch_cat(list(q_weight, k_weight, v_weight), dim = 1L)

    q_bias <- decoded_dict[[paste0(prefix, "self_attn.q_proj.bias")]]
    k_bias <- decoded_dict[[paste0(prefix, "self_attn.k_proj.bias")]]
    v_bias <- decoded_dict[[paste0(prefix, "self_attn.v_proj.bias")]]
    result[[paste0(block_prefix, "attn.qkv.bias")]] <-
      torch_cat(list(q_bias, k_bias, v_bias), dim = 1L)

    # Output projection
    result[[paste0(block_prefix, "attn.out.weight")]] <-
      decoded_dict[[paste0(prefix, "self_attn.o_proj.weight")]]
    result[[paste0(block_prefix, "attn.out.bias")]] <-
      decoded_dict[[paste0(prefix, "self_attn.o_proj.bias")]]

    # MLP norm
    result[[paste0(block_prefix, "mlp.norm.scale")]] <-
      decoded_dict[[paste0(prefix, "post_attention_layernorm.weight")]]

    # MLP router/gate
    result[[paste0(block_prefix, "mlp.gate.weight")]] <-
      decoded_dict[[paste0(prefix, "mlp.router.weight")]]
    result[[paste0(block_prefix, "mlp.gate.bias")]] <-
      decoded_dict[[paste0(prefix, "mlp.router.bias")]]

    # MLP experts (already decoded from MXFP4)
    result[[paste0(block_prefix, "mlp.mlp1_weight")]] <-
      decoded_dict[[paste0(prefix, "mlp.experts.gate_up_proj")]]
    result[[paste0(block_prefix, "mlp.mlp1_bias")]] <-
      decoded_dict[[paste0(prefix, "mlp.experts.gate_up_proj_bias")]]
    result[[paste0(block_prefix, "mlp.mlp2_weight")]] <-
      decoded_dict[[paste0(prefix, "mlp.experts.down_proj")]]
    result[[paste0(block_prefix, "mlp.mlp2_bias")]] <-
      decoded_dict[[paste0(prefix, "mlp.experts.down_proj_bias")]]
  }

  result
}

gptoss_decode_mxfp4 <- function(
  blocks,
  scales,
  dtype = torch_bfloat16(),
  rows_per_chunk = 16384L * 512L
) {
  scales <- scales$to(dtype = torch_int()) - 127L
  blocks_shape <- as.integer(blocks$shape)
  scales_shape <- as.integer(scales$shape)

  if (!identical(blocks_shape[-length(blocks_shape)], scales_shape)) {
    cli::cli_abort("{.arg blocks} and {.arg scales} shapes are incompatible.")
  }

  lut <- torch_tensor(gptoss_fp4_values, dtype = dtype, device = blocks$device)
  prefix_shape <- blocks_shape[seq_len(length(blocks_shape) - 2L)]
  g <- blocks_shape[length(blocks_shape) - 1L]
  b <- blocks_shape[length(blocks_shape)]
  rows_total <- prod(prefix_shape) * g

  blocks <- blocks$reshape(c(rows_total, b))
  scales <- scales$reshape(c(rows_total, 1L))
  out <- torch_empty(c(rows_total, b * 2L), dtype = dtype, device = blocks$device)

  for (r0 in seq.int(1L, rows_total, by = rows_per_chunk)) {
    r1 <- min(r0 + rows_per_chunk - 1L, rows_total)
    blk <- blocks[r0:r1, ]
    exp <- scales[r0:r1, ]

    idx_lo <- torch_bitwise_and(blk, 15L)$to(dtype = torch_long()) + 1L
    idx_hi <- blk$bitwise_right_shift(4L)$to(dtype = torch_long()) + 1L

    sub <- out[r0:r1, ]
    interleaved <- torch_stack(list(lut[idx_lo], lut[idx_hi]), dim = -1)$
      reshape(c(r1 - r0 + 1L, b * 2L))
    sub$copy_(interleaved)
    sub$ldexp_(exp)
  }

  out <- out$reshape(c(prefix_shape, g, b * 2L))
  out$view(c(prefix_shape, g * b * 2L))
}

#' GPT-OSS
#'
#' Initializes a GPT-OSS style model (OpenAI reference PyTorch architecture).
#'
#' @param num_hidden_layers Number of transformer blocks.
#' @param num_experts Number of routed experts in each MLP block.
#' @param experts_per_token Number of experts selected per token.
#' @param vocab_size Vocabulary size.
#' @param hidden_size Hidden dimension.
#' @param intermediate_size MLP intermediate size per expert.
#' @param swiglu_limit Clamp limit used in SwiGLU.
#' @param head_dim Attention head dimension.
#' @param num_attention_heads Number of attention heads.
#' @param num_key_value_heads Number of key/value heads (GQA).
#' @param sliding_window Sliding window size for alternating local-attention blocks.
#' @param initial_context_length Initial context length used by YaRN scaling.
#' @param rope_theta Rotary base.
#' @param rope_scaling_factor YaRN scaling factor.
#' @param rope_ntk_alpha YaRN NTK alpha.
#' @param rope_ntk_beta YaRN NTK beta.
#' @returns An initialized [torch::nn_module()].
#' @export
gptoss <- function(
  num_hidden_layers = 36L,
  num_experts = 128L,
  experts_per_token = 4L,
  vocab_size = 201088L,
  hidden_size = 2880L,
  intermediate_size = 2880L,
  swiglu_limit = 7.0,
  head_dim = 64L,
  num_attention_heads = 64L,
  num_key_value_heads = 8L,
  sliding_window = 128L,
  initial_context_length = 4096L,
  rope_theta = 150000.0,
  rope_scaling_factor = 32.0,
  rope_ntk_alpha = 1.0,
  rope_ntk_beta = 32.0
) {
  config <- gptoss_normalize_config(as.list(environment()))
  nn_gptoss_model(config)
}

#' @describeIn gptoss Initializes a gptoss model using a configuration defined in HF Hub.
#' @param identifier A string representing the model identifier in the Hugging Face model hub.
#' @param revision A string specifying the revision in the Hugging Face model hub.
#' @export
gptoss_from_config <- function(identifier = "openai/gpt-oss-20b", revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)
  config <- gptoss_normalize_config(as.list(config))
  do.call(gptoss, config)
}

#' @describeIn gptoss Initializes a gptoss model and loads pre-trained weights from HF Hub.
#' @export
gptoss_from_pretrained <- function(identifier = "openai/gpt-oss-20b", revision = "main") {
  with_device(device = "meta", {
    model <- gptoss_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- gptoss_hf_weights_remap(state_dict)
  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}
