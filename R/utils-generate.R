logit_transform <- function(temperature = 1, top_k = 50) {
  force(temperature)
  force(top_k)
  function(logits) {
    logits <- logits/temperature
    c(prob, ind) %<-% logits$topk(top_k)
    logits <- torch_full_like(logits, -Inf)$scatter_(-1, ind, prob)
    nnf_softmax(logits, dim = -1)
  }
}

#' @importFrom rlang %||%
generate_sample <- function(config = list(), logit_transform = NULL) {
  config$bos_token_id <- config$bos_token_id %||% 0
  config$eos_token_id <- config$eos_token_id %||% 0
  logit_transform <- logit_transform %||% logit_transform()
  function(model, idx, max_new_tokens = 100) {
    model$eval()
    for (i in seq_len(max_new_tokens)) {
      with_no_grad({
        logits <- model(idx + 1L)
      })
      logits <- logit_transform(logits[,-1,])
      id_next <- torch_multinomial(logits, num_samples = 1) - 1L

      if (id_next$item() == config$eos_token_id) {
        break
      }

      idx <- torch_cat(list(idx, id_next), dim = 2)
    }
    idx
  }
}
