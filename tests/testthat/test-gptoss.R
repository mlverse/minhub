test_that("gptoss tiny model can run forward pass", {
  model <- gptoss(
    num_hidden_layers = 2L,
    num_experts = 8L,
    experts_per_token = 2L,
    vocab_size = 128L,
    hidden_size = 64L,
    intermediate_size = 64L,
    swiglu_limit = 7.0,
    head_dim = 16L,
    num_attention_heads = 4L,
    num_key_value_heads = 2L,
    sliding_window = 8L,
    initial_context_length = 64L,
    rope_theta = 10000.0,
    rope_scaling_factor = 1.0,
    rope_ntk_alpha = 1.0,
    rope_ntk_beta = 32.0
  )

  idx <- torch_tensor(c(3L, 2L, 7L, 9L), dtype = torch_long())

  with_no_grad({
    out <- model(idx)
  })

  expect_equal(as.integer(out$shape), c(4L, 128L))
  expect_equal(as.character(out$dtype), "BFloat16")
})

test_that("gptoss-20b can generate a sensible continuation", {
  skip("Skipping GPT-OSS-20B generation test in CI")
  identifier <- "openai/gpt-oss-20b"
  model <- gptoss_from_pretrained(identifier)

  tok <- tok::tokenizer$from_pretrained(identifier)
  prompt <- "Hello world!"
  idx <- torch_tensor(tok$encode(prompt)$ids, dtype = torch_long())$view(c(1, -1))

  eos_token_id <- 200002L
  max_new_tokens <- 5L
  generated_ids <- integer()

  for (i in seq_len(max_new_tokens)) {
    with_no_grad({
      logits <- model(idx + 1L)[, -1, ]
    })

    id_next <- as.integer(torch_argmax(logits, dim = -1)$item()) - 1L
    if (id_next == eos_token_id) {
      break
    }

    generated_ids <- c(generated_ids, id_next)
    id_next_tensor <- torch_tensor(id_next, dtype = torch_long())$view(c(1, 1))
    idx <- torch_cat(list(idx, id_next_tensor), dim = 2)
  }

  generated_text <- tok$decode(generated_ids)
  full_text <- tok$decode(as.integer(idx))

  expect_gt(length(generated_ids), 4L)
  expect_gt(length(unique(generated_ids)), 3L)
  expect_true(grepl("[A-Za-z]{3,}", generated_text))
  expect_true(grepl("[[:space:]]", generated_text))
  expect_gt(nchar(full_text), nchar(prompt))
})
