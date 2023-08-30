test_that("Can create a llama model", {
  skip_on_ci() # this is too big for the github runners.
  model <- llama_from_pretrained("meta-llama/Llama-2-7b-chat-hf")
  model$to(dtype=torch_float32())
  model$eval()
  with_no_grad({
    pred <- model(torch_ones(1, 3, dtype="long") + 1L)
  })

  out <- pred[1, 1, 1:5]

  reference <- c(0.2226,  0.0299,  0.2729, -0.7919,  1.6164)
  expect_equal(as.numeric(out), reference, tolerance = 1e-4)
})

test_that("can create predictions", {
  skip_on_ci()

  model <- llama_from_pretrained("codellama/CodeLlama-7b-hf")
  model$to(dtype=torch_float32())
  model$eval()
  with_no_grad({
    pred <- model(torch_ones(1, 3, dtype="long") + 1L)
  })

  out <- pred[1, 1, 1:5]

  reference <- c(-7.1554, -13.0050,   5.4070,  -6.0711,  -5.8435)
  expect_equal(as.numeric(out), reference, tolerance = 1e-5)

})
