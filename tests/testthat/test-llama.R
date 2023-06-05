test_that("Can create a llama model", {
  model <- llama_from_pretrained("huggyllama/llama-7b")
  model$to(dtype=torch_float32())
  model$eval()
  with_no_grad({
    pred <- model(torch_ones(1, 3, dtype="long") + 1L)
  })

  out <- pred[1, 1, 1:5]
  reference <- c(-12.7782, -28.6373,   0.9082,  -6.1501,  -4.3769)
  expect_equal(as.numeric(out), reference, tolerance = 1e-5)
})
