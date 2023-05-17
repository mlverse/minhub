test_that("gptneox", {
  identifier <- "EleutherAI/pythia-70m"
  revision <- "495869e"
  model <- gptneox_from_pretrained(identifier, revision)
  tok <- tok::tokenizer$from_pretrained(identifier)
  model$eval()
  model$to(dtype = torch_float())
  model$eval()
  idx <- torch_tensor(tok$encode("Hello world ")$ids)$view(c(1, -1))
  with_no_grad({
    out <- model(idx + 1L)
  })

  reference <- c(1050.45031738281, 224.339889526367, 1047.935546875, 1045.73510742188, 1047.39111328125)
  result <- as.numeric(out[,-1,][,1:5])
  expect_equal(result, reference, tolerance = 1e-6)
})
