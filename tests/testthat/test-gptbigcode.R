test_that("default model nparam", {
  model <- gptbigcode()

  n_parameters <- sum(sapply(model$parameters, function(x) x$numel()))
  expect_equal(n_parameters, 150044160)
})

test_that("can run bigcode", {
  identifier <- "bigcode/gpt_bigcode-santacoder"
  # 150044160
  model <- gptbigcode_from_pretrained(identifier)
  model$to(dtype=torch_float())
  tok <- tok::tokenizer$from_pretrained(identifier)
  model$eval()
  idx <- torch_tensor(tok$encode("def sum(x, y):")$ids)$view(c(1, -1))
  with_no_grad({
    out <- model(idx + 1L)
  })

  reference <- c(7.76255893707275, 11.3832864761353, 14.9211025238037, 6.05046129226685, 8.51452255249023)
  result <- as.numeric(out[,-1,1:5])
  expect_equal(result, reference, tolerance = 1e-5)
})

