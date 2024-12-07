nn_gptbigcode_attention <- nn_module(
  initialize = function(n_head, n_embd, max_pos, pdrop) {
    self$n_head <- n_head
    self$n_embd <- n_embd
    self$d_head <- n_embd / n_head
    self$max_pos <- max_pos

    self$c_attn <- nn_linear(n_embd, n_embd + 2*self$d_head)
    self$c_proj <- nn_linear(n_embd, n_embd)

    self$attn_dropout <- nn_dropout(pdrop)
    self$resid_dropout <- nn_dropout(pdrop)
    self$register_bias()
  },
  forward = function(x) {
    c(b, t, h) %<-% x$shape

    c(q, k, v) %<-% self$c_attn(x)$
      split(c(self$n_embd, self$d_head, self$d_head), dim = -1)

    # q is [b, t, n_head*d_head]
    # we make it [b, t*n_head, d_head]
    q <- q$reshape(c(b, t*self$n_head, self$d_head))

    att <- torch_matmul(q, k$transpose(-2, -1)) * (1 / sqrt(k$size(-1)))
    att <- att$view(c(b, t, self$n_head, t))
    att <- att$masked_fill(self$bias[,,1:t, 1:t]$transpose(2,3) == 0, -Inf)
    att <- nnf_softmax(att, dim=-1)

    att <- self$attn_dropout(att)
    y <- torch_matmul(att$view(c(b, self$n_head*t, t)), v)$view(c(b, t, h))
    self$c_proj(y)
  },
  register_bias = function() {
    max_pos <- self$max_pos
    self$bias <- torch_ones(max_pos, max_pos)$
      bool()$
      tril()$
      view(c(1, 1, max_pos, max_pos)) |>
      nn_buffer(persistent = FALSE)
  },
  .load_from_state_dict = function(...) {
    super$.load_from_state_dict(...)
    self$register_bias()
  }
)

nn_gptbigcode_mlp <- nn_module(
  initialize = function(n_embd, pdrop, n_inter = 4*n_embd) {
    self$c_fc <- nn_linear(n_embd, n_inter)
    self$c_proj <- nn_linear(n_inter, n_embd)
    self$act <- nn_gelu(approximate = "tanh")
    self$dropout <- nn_dropout(pdrop)
  },
  forward = function(x) {
    x |>
      self$c_fc() |>
      self$act() |>
      self$c_proj() |>
      self$dropout()
  }
)

nn_gptbigcode_layer <- nn_module(
  initialize = function(n_embd, n_head, max_pos, pdrop) {
    self$ln_1 <- nn_layer_norm(n_embd)
    self$attn <- nn_gptbigcode_attention(n_head, n_embd, max_pos, pdrop)
    self$ln_2 <- nn_layer_norm(n_embd)
    self$mlp <- nn_gptbigcode_mlp(n_embd, pdrop)
  },
  forward = function(x) {
    x <- x + self$attn(self$ln_1(x))
    x <- x + self$mlp(self$ln_2(x))
    x
  }
)

nn_gptbigcode_model <- nn_module(
  initialize = function(vocab_size, n_embd, n_head, n_layer, max_pos, pdrop) {
    self$transformer <- nn_module_dict(list(
      wte = nn_embedding(vocab_size, n_embd),
      wpe = nn_embedding(max_pos, n_embd),
      drop = nn_dropout(pdrop),
      h = nn_sequential(!!!map(
        1:n_layer,
        \(x) nn_gptbigcode_layer(n_embd, n_head, max_pos, pdrop)
      )),
      ln_f = nn_layer_norm(n_embd)
    ))
    self$lm_head <- nn_linear(n_embd, vocab_size, bias = FALSE)
  },
  forward = function(idx) {
    c(b, t) %<-% idx$shape
    pos <- torch_arange(1, t, dtype = "int", device = idx$device)$unsqueeze(1)

    tok_emb <- self$transformer$wte(idx)
    pos_emb <- self$transformer$wpe(pos)

    x <- self$transformer$drop(tok_emb + pos_emb)
    x <- self$transformer$h(x)
    x <- self$transformer$ln_f(x)
    self$lm_head(x)
  }
)

#' GPT BigCode
#'
#' Initializes a BigCode model
#'
#' @param vocab_size An integer indicating the size of the vocabulary or the number
#'   of unique tokens in the input data.
#' @param n_embd An integer specifying the dimensionality of the embedding vectors.
#' @param n_head An integer representing the number of attention heads in the
#'   multi-head attention mechanism.
#' @param n_layer An integer indicating the number of layers in the deep learning model.
#' @param max_pos An integer specifying the maximum position encoding value or
#'  the maximum sequence length.
#' @param pdrop Dropout probability the attention, residual and embeddings dropout.
#' @param identifier A string representing the identifier or name of the pre-trained
#'  model in the Hugging Face model hub.
#' @param revision A string specifying the revision or version of the pre-trained
#'  model in the Hugging Face model hub.
#'
#' @returns An initialized [torch::nn_module()].
#' @export
gptbigcode <- function(vocab_size = 50257, n_embd = 768, n_head = 12, n_layer = 12,
                       max_pos = 1024, pdrop = 0.1) {
  nn_gptbigcode_model(vocab_size, n_embd, n_head, n_layer, max_pos, pdrop)
}

#' @describeIn gptbigcode Initializes a `gptbigcode` from a config file from HF hub
#' @export
gptbigcode_from_config <- function(identifier, revision = "main") {
  path <- hfhub::hub_download(identifier, "config.json", revision = revision)
  config <- jsonlite::fromJSON(path)

  if (config$model_type != "gpt_bigcode")
    cli::cli_abort(gettext(
      x = "{.arg config$model_type} must be {.val gpt_bigcode}.",
      i = "Got {.val {config$model_type}}",
      domain = "R-minhub"
    ))

  if (!config$multi_query)
    cli::cli_abort(gettext("Must use {.arg config$multi_query} but got {.val FALSE}",
                   domain = "R-minhub"))

  dropouts <- config[c("attn_pdrop", "resid_pdrop", "embd_pdrop")]
  if (length(unique(dropouts)) != 1)
    cli::cli_abort(gettext(
      x = "All dropout must be equal.",
      i = "Got {.val {names(dropouts)}} respectively {.val {dropouts}}",
      domain = "R-minhub"
    ))
  else
    pdrop <- unique(dropouts)


  if (config$layer_norm_eps != 1e-5)
    cli::cli_abort(gettext(
      x = "{.arg config$layer_norm_eps} must be 1e-5, got {.val {config$layer_norm_eps}}",
      domain = "R-minhub"
    ))

  # remap HF config attributes to minhub configurations
  vocab_size <- config$vocab_size
  n_embd     <- config$n_embd
  n_head     <- config$n_head
  n_layer    <- config$n_layer
  max_pos    <- config$n_positions
  pdrop      <- unlist(pdrop)

  gptbigcode(vocab_size, n_embd, n_head, n_layer, max_pos, pdrop)
}

#' @describeIn gptbigcode Initializes a `gptbigcode` and loads pre-trained weights from HF Hub
#' @export
gptbigcode_from_pretrained <- function(identifier, revision = "main") {
  with_device(device="meta", {
    model <- gptbigcode_from_config(identifier, revision)
  })

  state_dict <- hf_state_dict(identifier, revision)
  # some state dicts don't include the lm_head as it's the same as the
  # token embedding weights.
  if (is.null(state_dict$lm_head.weight)) {
    state_dict$lm_head.weight <- state_dict$transformer.wte.weight
  }

  model$load_state_dict(state_dict, .refer_to_state_dict = TRUE)
  model
}
