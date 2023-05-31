test_that("complete workflow: load model, tokenize, predict", {
  identifier <- "gpt2"
  revision <- "e7da7f2"

  # downloads https://huggingface.co/gpt2/blob/main/pytorch_model.bin, which needs to be converted to the new (zip) format
  # temporary workaround: load and save in current PyTorch, save to
  # .cache/huggingface/hub/models--gpt2/blobs/2d19e321961949b7f761cdffefff32c0-66
  # and check symlink at
  # .cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8/pytorch-model.bin

  model <- gpt2_from_pretrained(identifier, revision)
  tok <- tok::tokenizer$from_pretrained(identifier) # will download matching tokenizer from https://huggingface.co/gpt2/resolve/e7da7f2/tokenizer.json
  model$eval()
  idx <- torch_tensor(tok$encode("Hello world ")$ids)$view(c(1, -1))
  with_no_grad({
    logits <- model(idx + 1L)
  })
  as.numeric(logits[1, -1, 1:5]) # [1] -28.26509 -27.30587 -29.63981 -30.19297 -29.37851

  # this is what happens in Python
  # logits are different
# import transformers
# import torch
# from transformers import AutoTokenizer
# from transformers import GPT2LMHeadModel
# model_name = "GPT2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# idx = "Hello world "
# encoding = tokenizer.encode(idx)
# model = GPT2LMHeadModel.from_pretrained(model_name)
# pred = model(torch.tensor(encoding))["logits"]
# pred.shape # torch.Size([3, 50257])
# pred[-1, 0:5] # tensor([-53.2292, -55.5639, -58.5087, -57.6649, -59.0031], grad_fn=<SliceBackward0>)

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

