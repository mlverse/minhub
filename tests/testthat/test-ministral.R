test_that("Can create a ministral model from Mistral-7B", {

  skip_on_ci() # too large for CI runners

  model <- ministral_from_pretrained("mistralai/Mistral-7B-v0.1")
  
  model$to(dtype = torch_float32())
  model$eval()

  # Input tokens (R uses 1-based embedding indexing, so add 1 to Python indices)
  # This corresponds to Python input_ids = [2, 3, 4, 5, 6]
  with_no_grad({
    pred <- model(torch_tensor(matrix(c(3L, 4L, 5L, 6L, 7L), nrow = 1)))
  })

  # Reference from Python transformers MistralForCausalLM
  # output[0, 0, :5] for input_ids [[2, 3, 4, 5, 6]]
  out <- pred[1, 1, 1:5]
  reference <- c(1.4050, 7.4602, -0.5856, 0.5367, 1.4043)
  expect_equal(as.numeric(out), reference, tolerance = 1e-4)
})

test_that("Can create a ministral model with custom config", {
  model <- ministral(
    vocab_size = 1000,
    n_embd = 256,
    n_inter = 512,
    n_head = 8,
    n_kv_head = 2,
    head_dim = 32,
    n_layer = 2,
    max_pos = 512
  )

  input_ids <- torch_randint(1, 1000, c(2, 10), dtype = torch_long())
  with_no_grad({
    out <- model(input_ids)
  })

  expect_equal(out$shape, c(2, 10, 1000))
})

test_that("Can generate text with ministral", {
  skip_on_ci() # too large for CI runners
  skip_if_not_installed("tok")

  model <- ministral_from_pretrained("mistralai/Mistral-7B-v0.1")
  model$eval()

  tokenizer <- tok::tokenizer$from_pretrained("mistralai/Mistral-7B-v0.1")

  # Generation parameters
  prompt <- "The capital of France is"
  max_new_tokens <- 20
  temperature <- 0.7
  top_k <- 50
  eos_token_id <- 2

  # Encode prompt (tok returns 0-indexed token ids)
  encoded <- tokenizer$encode(prompt)
  # Add 1 for R's 1-based embedding indexing
  idx <- torch_tensor(encoded$ids, dtype = torch_long())$unsqueeze(1) + 1L

  generated_ids <- c()
  prev_text <- ""

  cat("\n", prompt, sep = "")
  for (i in seq_len(max_new_tokens)) {
    with_no_grad({
      logits <- model(idx)
    })

    # Get logits for last position
    logits <- logits[1, -1, ]

    # Apply temperature and top-k
    logits <- logits / temperature
    c(topk_values, topk_indices) %<-% logits$topk(top_k)
    logits <- torch_full_like(logits, -Inf)
    logits$scatter_(1, topk_indices, topk_values)

    # Sample
    probs <- nnf_softmax(logits, dim = -1)
    id_next <- torch_multinomial(probs, num_samples = 1L)
    id_next_val <- as.integer(id_next) - 1L

    if (id_next_val == eos_token_id) break

    generated_ids <- c(generated_ids, id_next_val)

    # Stream output: decode full sequence and print only new characters
    full_text <- tokenizer$decode(generated_ids)
    new_text <- substr(full_text, nchar(prev_text) + 1, nchar(full_text))
    cat(new_text)
    prev_text <- full_text

    idx <- torch_cat(list(idx, id_next$unsqueeze(1)), dim = 2)
  }
  cat("\n")

  # Basic check - we generated something
  expect_true(length(generated_ids) > 0)
})
