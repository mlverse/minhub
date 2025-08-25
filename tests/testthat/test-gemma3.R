test_that("Can create a gemma3 model", {
  skip_on_ci()
  model <- gemma3_from_pretrained("google/gemma-3-270m")
  model$to(dtype=torch_float64())
  model$eval()
  with_no_grad({
    pred <- model(torch_ones(1, 3, dtype="long") + 1L)
  })

  out <- pred[1, 1, 1:5]
  expect_true(all(is.finite(as.numeric(out))))
})

test_that("gemma3 generation", {
  skip_on_ci()

  identifier <- "google/gemma-3-270m"
  model <- gemma3_from_pretrained(identifier)
  model$to(dtype=torch_float64())
  tok <- tok::tokenizer$from_pretrained(identifier)

  prompt <- "Give me a short introduction to large language models."

  chat_template <- function(user_text) {
    glue::glue("<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n")
  }

  idx <- torch_tensor(tok$encode(chat_template(prompt))$ids)$view(c(1, -1))

  path <- hfhub::hub_download(identifier, "config.json", revision = "main")
  config <- jsonlite::fromJSON(path)

  sampler <- generate_sample(config, logit_transform())
  s <- sampler(model, idx)

  cat(paste(tok$decode(as.integer(s$to(device="cpu"))), collapse  = ""))

})
