test_that("can create predictions", {
  skip_on_ci()

  model <- codellama_from_pretrained("codellama/CodeLlama-7b-hf")
  model$to(dtype=torch_float32())
  model$eval()
  with_no_grad({
    pred <- model(torch_ones(1, 3, dtype="long") + 1L)
  })

  out <- pred[1, 1, 1:5]

  reference <- c(-7.1554, -13.0050,   5.4070,  -6.0711,  -5.8435)
  expect_equal(as.numeric(out), reference, tolerance = 1e-5)

})
