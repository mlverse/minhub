test_that("complete workflow: load model, tokenize, predict", {
  identifier <- "gpt2"
  revision <- "e7da7f2"

  model <- gpt2_from_pretrained(identifier, revision)

  par_count <- sum(purrr::map_int(model$parameters, ~.x$numel()))
  expect_equal(par_count, 124439808) # from HF

  tok <- tok::tokenizer$from_pretrained(identifier)
  idx <- torch_tensor(tok$encode("Hello world ")$ids)$view(c(1, -1))
  expect_equal(as.integer(idx), c(15496, 995, 220)) # from HF

  model$eval()
  with_no_grad({
    logits <- model(idx + 1L)
  })

  reference <- c(-53.2292, -55.5639, -58.5087, -57.6649, -59.0031)
  value <- as.numeric(logits[1, -1, 1:5])

  expect_equal(value, reference, tolerance = 1e-5)

})

test_that("can generate samples", {
  identifier <- "gpt2"
  revision <- "e7da7f2"
  model <- gpt2_from_pretrained(identifier, revision)
  tok <- tok::tokenizer$from_pretrained(identifier)
  model$eval()
  idx <- torch_tensor(tok$encode("No duty is imposed on the rich, rights of the poor is a hollow phrase ... Enough languishing in custody. Equality")$ids)$view(c(1, -1))
  prompt_length <- idx$size(-1)
  for (i in 1:30) {
    with_no_grad({
      logits <- model(idx + 1L)
    })
    last_logits <- logits[ , -1, ]
    c(prob, ind) %<-% last_logits$topk(50)
    last_logits <- torch_full_like(last_logits, -Inf)$scatter_(-1, ind, prob)
    probs <- nnf_softmax(last_logits, dim = -1)
    id_next <- torch_multinomial(probs, num_samples = 1) - 1L
    if (id_next$item() == 0) {
      break
    }
    idx <- torch_cat(list(idx, id_next), dim = 2)
  }
  # tok$decode(as.integer(idx))
  expect_lte(idx$size(-1), prompt_length + 30)
})


test_that("lm_head$weight is tied to transformer$wte$weight", {
  identifier <- "gpt2"
  revision <- "e7da7f2"
  model <- gpt2_from_pretrained(identifier, revision)
  wte <- model$transformer$wte$weight
  lm_head <- model$lm_head$weight
  expect_equal(as.numeric(torch_mean(wte)), as.numeric(torch_mean(lm_head)))
})

test_that("can execute gpt2 after moving to different device", {
  skip_on_ci()
  skip_if(!torch::backends_mps_is_available())

  model <- gpt2(vocab_size = 1000)
  model$to(device = "mps")

  x <- torch_tensor(sample.int(100, size = 100), device = "mps")$view(c(1, -1))

  expect_error({
    with_no_grad({
      out <- model(x)
    })
  }, regexp = NA)

})

test_that("wrong identifier raise an error", {
  identifier <- "Qwen/Qwen2.5-Coder-0.5B"
  revision <- "main"
  expect_error(
    model <- gpt2_from_pretrained(identifier, revision)    ,
    regexp = " must be \"gpt2\", got \"qwen2\"")

})


