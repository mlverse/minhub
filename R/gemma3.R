# References:
# - https://huggingface.co/docs/transformers/model_doc/gemma3
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py
# - https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/standalone-gemma3.ipynb

#' @noRd
#' @importFrom dotty .
#' @importFrom purrr map
#' @import torch
NULL

nn_gemma3_rmsnorm <- nn_module(
  initialize = function(hidden_size, eps = 1e-6, add_unit_offset = TRUE) {
    self$eps <- eps
    self$add_unit_offset <- add_unit_offset
    self$weight <- nn_parameter(torch_zeros(hidden_size))
  },
  forward = function(x) {
    # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output <- self$norm(x$float())
    output <- if (self$add_unit_offset) {
      output * (1 + self$weight$float())
    } else {
      output * self$weight$float()
    }
    output$type_as(x)
  },
  norm = function(x) {
    x * torch_rsqrt(x$pow(2)$mean(-1, keepdim = TRUE) + self$eps)
  }
)

gemma3_rotate_half <- function(x) {
  .[x1, x2] <- torch_split(x, x$size(-1) / 2, -1)
  torch_cat(list(-x2, x1), dim = -1)
}

nn_gemma3_rotary_embedding <- nn_module(
  initialize = function(d_head, max_pos, base = 10000) {
    self$d_head <- d_head
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

      inv_freq <- 1 / self$base ^ (torch_arange(0, self$d_head - 1, step = 2) / self$d_head)

      freqs <- torch_arange(start = 0, end = self$max_pos - 1)$
        float()$
        outer(inv_freq)$
        view(c(1, 1, self$max_pos, self$d_head / 2))

      emb <- torch_cat(list(freqs, freqs), dim = -1)

      self$cos <- nn_buffer(emb$cos(), persistent = FALSE)
      self$sin <- nn_buffer(emb$sin(), persistent = FALSE)
    }
    list(self$cos[,,1:t,], self$sin[,,1:t,])
  },
  forward = function(x) {
    .[b, n_head, t, d_head] <- x$shape
    .[cos, sin] <- self$cached_embeddings(t)

    # rotary embeddings are applied only to the first `n_rot` dims of x
    x * cos + gemma3_rotate_half(x) * sin
  }
)

nn_gemma3_attention_mask <- nn_module(
  initialize = function(max_pos, sliding_window = NULL) {
    full <- torch_ones(max_pos, max_pos, dtype = "bool")

    mask <- if (is.null(sliding_window)) {
      # global mask
      full$triu(diagonal = 1)
    } else {
      # local mask
      full$triu(diagonal = 1) | full$triu(diagonal = sliding_window)$t()
    }

    self$mask <- nn_buffer(
      mask$view(c(1, 1, max_pos, max_pos)),
      persistent = FALSE
    )
  },
  forward = function(x) {
    .[b, n_head, t, d_head] <- x$shape
    x$masked_fill(self$mask[,,1:t,1:t], -Inf)
  }
)

nn_gemma3_attention <- nn_module(
  initialize = function(n_embd, n_head, d_head, max_pos, rope_base, mask) {
    self$n_head <- n_head
    self$d_head <- d_head

    self$k_proj <- nn_linear(n_embd, d_head, bias = FALSE)
    self$q_proj <- nn_linear(n_embd, d_head * n_head, bias = FALSE)
    self$v_proj <- nn_linear(n_embd, d_head, bias = FALSE)

    self$q_norm <- nn_gemma3_rmsnorm(d_head, eps=1e-6)
    self$k_norm <- nn_gemma3_rmsnorm(d_head, eps=1e-6)

    self$o_proj <- nn_linear(d_head * n_head, n_embd, bias = FALSE)
    self$rotary <- nn_gemma3_rotary_embedding(d_head, max_pos, rope_base)

    self$mask <- mask
  },
  forward = function(x) {
    .[b, t, h] <- x$shape

    q <- self$q_proj(x) # (b, t, n_head * d_head)
    k <- self$k_proj(x) # (b, t, d_head)
    v <- self$v_proj(x) # (b, t, d_head)

    q <- q$view(c(b, t, self$n_head, self$d_head))$transpose(2, 3)
    k <- k$view(c(b, t, 1, self$d_head))$transpose(2, 3)
    v <- v$view(c(b, t, 1, self$d_head))$transpose(2, 3)
    
    q <- self$q_norm(q)
    k <- self$k_norm(k)

    q <- self$rotary(q)
    k <- self$rotary(k)

    # 1L is the group size for the small model
    k <- k$repeat_interleave(1L, dim = 2L)
    v <- v$repeat_interleave(1L, dim = 2L)

    q <- q * (1 / sqrt(256))

    att <- torch_matmul(q, k$transpose(-2, -1))
    att <- self$mask(att)
    att <- nnf_softmax(att, dim = -1)$to(dtype = v$dtype)

    o <- torch_matmul(att, v)$transpose(2, 3)$contiguous()$view(c(b, t, -1))
    self$o_proj(o)
  },
  cached_bias = function() {
    # causal mask to ensure that attention is only applied to the left in the
    # input sequence
    self$bias <- torch_ones(self$max_pos, self$max_pos)$bool()$tril()$view(c(
      1,
      1,
      self$max_pos,
      self$max_pos
    )) |>
      nn_buffer(persistent = FALSE)

    self$masked_bias <- nn_buffer(torch_scalar_tensor(-Inf), persistent = FALSE)
  }
)

nn_gemma3_mlp <- nn_module(
  initialize = function(n_embd, n_inter) {
    self$gate_proj <- nn_linear(n_embd, n_inter, bias = FALSE)
    self$down_proj <- nn_linear(n_inter, n_embd, bias = FALSE)
    self$up_proj <- nn_linear(n_embd, n_inter, bias = FALSE)
    self$act <- nn_gelu(approximate = "tanh")
  },
  forward = function(x) {
    self$down_proj(self$act(self$gate_proj(x)) * self$up_proj(x))
  }
)

nn_gemma3_layer <- nn_module(
  initialize = function(
    n_embd,
    d_head,
    n_inter,
    n_head,
    max_pos,
    rmsnorm_eps,
    rope_base,
    att_mask
  ) {
    self$input_ln <- nn_gemma3_rmsnorm(n_embd, rmsnorm_eps)
    self$pst_att_ln <- nn_gemma3_rmsnorm(n_embd, rmsnorm_eps)
    self$pre_mlp_ln <- nn_gemma3_rmsnorm(n_embd, rmsnorm_eps)
    self$pst_mlp_ln <- nn_gemma3_rmsnorm(n_embd, rmsnorm_eps)

    self$attn <- nn_gemma3_attention(
      n_embd = n_embd,
      n_head = n_head,
      d_head = d_head,
      max_pos = max_pos,
      rope_base = rope_base,
      mask = att_mask
    )
    self$mlp <- nn_gemma3_mlp(n_embd, n_inter)
  },
  forward = function(x) {
    x <- x + self$pst_att_ln(self$attn(self$input_ln(x)))
    x + self$pst_mlp_ln(self$mlp(self$pre_mlp_ln(x)))
  }
)

nn_gemma3_model <- nn_module(
  initialize = function(
    vocab_size,
    n_embd,
    n_head,
    d_head,
    n_inter,
    layer_types,
    max_pos,
    sliding_window,
    rope_base,
    rmsnorm_eps
  ) {
    self$n_embd <- n_embd
    self$wte <- nn_embedding(vocab_size, n_embd)

    self$mask_global <- nn_gemma3_attention_mask(max_pos)
    self$mask_local <- nn_gemma3_attention_mask(
      max_pos,
      sliding_window = sliding_window
    )

    layers <- list(
      local = \(...) {
        nn_gemma3_layer(
          att_mask = self$mask_local,
          rope_base = rope_base["local"],
          ...
        )
      },
      global = \(...) {
        nn_gemma3_layer(
          att_mask = self$mask_global,
          rope_base = rope_base["global"],
          ...
        )
      }
    )

    self$blocks <- nn_sequential(
      !!!map(
        layer_types,
        function(type) {
          layers[[type]](
            n_embd = n_embd,
            d_head = d_head,
            n_inter = n_inter,
            n_head = n_head,
            max_pos = max_pos,
            rmsnorm_eps = rmsnorm_eps
          )
        }
      )
    )

    self$ln_f <- nn_gemma3_rmsnorm(n_embd, eps = 1e-6)
    self$lm_head <- nn_linear(n_embd, vocab_size, bias = FALSE)
  },
  forward = function(idx) {
    x <- self$wte(idx) * sqrt(self$n_embd)
    x <- self$blocks(x)
    x <- self$ln_f(x)
    self$lm_head(x)
  }
)

#' Gemma3
#'
#' @export
gemma3 <- function(
  vocab_size = 262144,
  n_embd = 640,
  n_head = 4,
  d_head = 256,
  n_inter = 2048,
  max_pos = 32768,
  sliding_window = 512,
  rope_base = c(local = 1e4, global = 1e6),
  layer_types = rep(c(rep("local", 5), "global"), times = 3),
  rmsnorm_eps = 1e-6
) {
  nn_gemma3_model(
    vocab_size = vocab_size,
    n_embd = n_embd,
    n_head = n_head,
    d_head = d_head,
    n_inter = n_inter,
    layer_types = layer_types,
    max_pos = max_pos,
    sliding_window = sliding_window,
    rope_base = rope_base,
    rmsnorm_eps = rmsnorm_eps
  )
}

#' @describeIn gemma3 Initializes a gemma3 model using a configuration defined in HF Hub
#' @export
gemma3_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  if (config$model_type != "gemma3_text") {
    cli::cli_abort(c(
      "{.arg config$model_type} must be {.val gemma3_text}, got {.val {config$model_type}}",
    ))
  }

  if (config$hidden_activation != "gelu_pytorch_tanh") {
    cli::cli_abort(c(
      x = "Unsupported {.arg config$hidden_activation}: {.val {config$hidden_activation}}",
      i = "Currently only {.val gelu_pytorch_tanh} is supported."
    ))
  }

  # remap HF config attributes to minhub configurations
  vocab_size <- config$vocab_size
  n_embd <- config$hidden_size
  n_inter <- config$intermediate_size
  n_head <- config$num_attention_heads
  d_head <- config$head_dim
  max_pos <- config$max_position_embeddings
  sliding_window <- config$sliding_window
  rmsnorm_eps <- config$rms_norm_eps # unlike most llama models, we also havve different values for this
  rope_base <- c(
    local = config$rope_local_base_freq,
    global = config$rope_theta
  )
  layer_types <- ifelse(
    config$layer_types == "sliding_attention",
    "local",
    "global"
  )

  gemma3(
    vocab_size = vocab_size,
    n_embd = n_embd,
    n_head = n_head,
    n_inter = n_inter,
    max_pos = max_pos,
    sliding_window = sliding_window,
    rope_base = rope_base,
    layer_types = layer_types,
    rmsnorm_eps = rmsnorm_eps
  )
}

#' @describeIn gemma3 Initializes the gemma3 model and load pre-trained weights from HF hub.
#' @export
gemma3_from_pretrained <- function(identifier = "google/gemma-3-270m", revision = "main") {
  with_device(device = "cpu", {
    model <- gemma3_from_config(identifier, revision)
  })
  state_dict <- hf_state_dict(identifier, revision)
  state_dict <- gemma3_hf_weights_remap(state_dict)

  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}

gemma3_hf_weights_remap <- function(state_dict) {
  nms <- names(state_dict)
  nms <- gsub("^model\\.", "", nms)
  nms <- gsub("^layers", "blocks", nms)
  nms <- gsub("self_attn", "attn", nms)
  nms <- gsub("input_layernorm", "input_ln", nms)
  nms <- gsub("post_attention_layernorm", "pst_att_ln", nms)
  nms <- gsub("pre_feedforward_layernorm", "pre_mlp_ln", nms)
  nms <- gsub("post_feedforward_layernorm", "pst_mlp_ln", nms)
  nms <- gsub("embed_tokens", "wte", nms)
  nms <- gsub("^norm", "ln_f", nms)
  names(state_dict) <- nms
  state_dict["lm_head.weight"] <- state_dict["wte.weight"]
  state_dict
}
