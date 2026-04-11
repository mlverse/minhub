test_that("Can create a qwen3 model from Qwen3.5-0.8B", {

  skip_on_ci() # too large for CI runners

  model <- qwen3_from_pretrained("Qwen/Qwen3.5-0.8B")

  model$to(dtype = torch_float32())
  model$eval()

  # Input tokens: R uses 1-based indexing, so add 1 to Python indices
  # This corresponds to Python input_ids = [1, 2, 3, 4, 5]
  with_no_grad({
    pred <- model(torch_tensor(matrix(2:6, nrow = 1), dtype = torch_long()))
  })

  # Reference from Python transformers Qwen3_5ForCausalLM (float32)
  # output[0, 0, :5] for input_ids [[1, 2, 3, 4, 5]]
  out <- pred[1, 1, 1:5]
  reference <- c(3.3154, 6.2008, 4.0241, 2.3807, 1.2581)
  expect_equal(as.numeric(out), reference, tolerance = 1e-3)
})

test_that("Can create a qwen3 model with custom config", {
  model <- qwen3(
    vocab_size = 1000,
    n_embd = 64,
    n_inter = 128,
    n_head = 2,
    n_kv_head = 1,
    head_dim = 32,
    n_layer = 4,
    max_pos = 1024,
    layer_types = c("linear_attention", "linear_attention",
                    "linear_attention", "full_attention"),
    n_k_heads = 4,
    n_v_heads = 4,
    k_head_dim = 16,
    v_head_dim = 16
  )

  input_ids <- torch_randint(1, 1000, c(2, 10), dtype = torch_long())
  with_no_grad({
    out <- model(input_ids)
  })

  expect_equal(out$shape, c(2, 10, 1000))
})
