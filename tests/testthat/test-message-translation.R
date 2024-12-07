identifier <- "Qwen/Qwen2.5-Coder-0.5B"
revision <- "8123ea2"

test_that("gpt2 R-level error messages are correctly translated in FR", {
  # skip on ubuntu cuda as image as no FR lang installed
  skip_if(torch::cuda_is_available() && grepl("linux-gnu", R.version$os))
  # skip on MAC M1
  skip_if(torch::backends_mps_is_available() && R.version$arch == "aarch64")
  withr::with_language(lang = "fr",
                       expect_error(
                         model <- gpt2_from_pretrained(identifier, revision),
                         regexp = " doit être \"gpt2\", or elle est à \"qwen2\"",
                         fixed = TRUE
                       )
  )
})

test_that("gptbigcode R-level error messages are correctly translated in FR", {
  # skip on ubuntu cuda as image as no FR lang installed
  skip_if(torch::cuda_is_available() && grepl("linux-gnu", R.version$os))
  # skip on MAC M1
  skip_if(torch::backends_mps_is_available() && R.version$arch == "aarch64")

  withr::with_language(lang = "fr",
                       expect_error(
                         model <- gptbigcode_from_pretrained(identifier, revision),
                         regexp = " doit être \"gpt_bigcode\".\nOr elle est à \"qwen2\"",
                         fixed = TRUE
                       )
  )
})
