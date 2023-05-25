test_that("can load pre-trained gpt2", {
  identifier <- "gpt2"
  revision <- "e7da7f2"

  # downloads https://huggingface.co/gpt2/blob/main/pytorch_model.bin, which needs to be converted to the new (zip) format
  # temporary workaround: load and save in current PyTorch, save to
  # .cache/huggingface/hub/models--gpt2/blobs/2d19e321961949b7f761cdffefff32c0-66
  # and check symlink at
  # .cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8/pytorch-model.bin

  model <- gpt2_from_pretrained(identifier, revision)
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


test_that("initialization works as expected", {
  model <- gpt2()
})

