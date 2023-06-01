nn_t5_layer_norm <- nn_module(
  initialize = function(normalized_shape, eps = 1e-6) {
    self$weight <- nn_parameter(torch_ones(hidden_size))
    self$eps <- eps
  },
  forward = function(x) {
    variance <- x$to(dtype="float32")$pow(2)$mean(dim = -1, keepdim = TRUE)
    x <- x * torch_rsqrt(variance + self$eps)

    self$weight * x$to(self$weight$dtype)
  }
)

nn_t5_position_embedding <- nn_module(
  initialize = function(n_buckets = 32, max_distance = 128, n_head) {
    self$n_buckets <- n_buckets
    self$max_distance <- max_distance
    self$embedding <- nn_embedding(n_buckets, n_head)

    self$cached_rel_pos()
  },
  cached_rel_pos = function(max_pos) {
    exact <- self$num_buckets / 2

    pos <- torch_arange(0, max_pos)
    rel_pos <- pos$unsqueeze(1) - pos$unsqueeze(-1)
    rel_pos <- -torch_min(relative_positions, other = 0)

    # buckets boundaries
    b <- nn_buffer(persistent = FALSE, torch_cat(list(
      -Inf,
      # the first half of the buckets are exact
      torch_arange(1, exact-1),
      # the second half uses an exponential increasing interval time
      torch_logspace(
        start = log(exact),
        end = log(self$max_distance),
        base = exp(1),
        steps = exact
      )
    )))
    rel_pos <- torch_bucketize(rel_pos, b)

    self$rel_pos <- nn_buffer(rel_pos, persistent = FALSE)
  },
  .load_from_state_dict = function(...) {
    super$.load_from_state_dict(...)
    self$cached_rel_pos()
  },
  forward = function(x) {
    c(b, nh, t, ed) %<-% x$shape
    rel_pos <- self$rel_pos[1:t, 1:t] # (t, t, nhead)
    self$embedding(rel_pos)$permute(c(3,1,2))$unsqueeze(1) # (1, t, t, nhead)
  }
)

nn_t5_attention <- nn_module(
  initialize = function(n_head, n_embd, max_pos, pdrop) {
    # unlike the other transformer implementations, the qkv projections
    # are defined separedtely
    self$c_q <- nn_linear(n_embd, n_embd * n_head / 3, bias = FALSE)
    self$c_k <- nn_linear(n_embd, n_embd * n_head / 3, bias = FALSE)
    self$c_v <- nn_linear(n_embd, n_embd * n_head / 3, bias = FALSE)
  },
  forward = function(x, pos_embd) {
    c(b, t, h) %<-% x$shape

    q <- self$c_q(x)$view(c(b, t, self$n_head, -1))
    k <- self$c_k(x)$view(c(b, t, self$n_head, -1))
    v <- self$c_v(x)$view(c(b, t, self$n_head, -1))

    att <- torch_matmul(q, k$transpose(-2, -1)) * (1 / sqrt(k$size(-1)))
    att <- att$masked_fill(self$bias[,,1:t, 1:t] == 0, self$masked_bias)
    att <- nnf_softmax(att, dim=-1)
  }
)

nn_t5_block <- nn_module(
  initialize = function() {
    self$position_embedding <- nn_t5_position_embedding()
  },
  forward = function(x) {
    pos_embd <- self$position_emebdding(x)

  }
)

nn_t5_stack <- nn_module(
  initialize = function() {

  },
  forward = function(x) {

  }
)

nn_t5_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_head, n_layer, max_pos) {
    self$wte <- nn_embedding(vocab_size, n_embd)
    self$decoder <- nn_t5_stack()
  },
  forward = function(idx) {
    x <- self$wte(idx)

  }
)

