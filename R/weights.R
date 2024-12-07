#' Loads, and possibly downloads HF Hub models
#' @param identifier Repository id from the model
#' @param revision Revision to download from (eg tags, branches or commit hashes)
#' @importFrom hfhub hub_download WEIGHTS_NAME WEIGHTS_INDEX_NAME
#' @export
hf_state_dict <- function(identifier, revision = "main") {

  state_dict <- try(state_dict_safetensors(identifier, revision))
  if (!inherits(state_dict, "try-error")) {
    return(state_dict)
  }

  err <- NULL
  # try downloading the weights from the pytorch_model.bin path and save error
  # if any happened
  weights_path <- tryCatch({
    hub_download(identifier, WEIGHTS_NAME(), revision = revision)
  }, error = function(err) {
    err <<- err
  })

  # if err is not null it probably means that the weights are sharded in the
  # repository, thus we look for the index.
  if (!is.null(err)) {
    # we now try looking for the index - if it doesn't exist it means the repo
    # doesn't contain any model or there's some connection problem.
    # in this case we want to raise an error showing the two urls that we tried.
    # we also prefer showing the stack trace from the first path.
    index_path <- tryCatch({
      hub_download(identifier, WEIGHTS_INDEX_NAME(), revision = revision)
    }, error = function(e) {
      cli::cli_abort(gettext(
        x = "Error downloading weights from {.val {c(WEIGHTS_NAME(), WEIGHTS_INDEX_NAME())}}",
        i = "Traceback below shows the error when trying to download {.val {WEIGHTS_NAME()}}",
        domain = "R-minhub"
      ), parent = err)
    })

    filenames <- unique(unlist(jsonlite::fromJSON(index_path)$weight_map))
    weights_path <- sapply(filenames, function(fname) {
      hub_download(identifier, fname, revision = revision)
    })
    names(weights_path) <- NULL
  }
  do.call("c", lapply(weights_path, torch::load_state_dict))
}

SAFETENSORS_NAME <- function() {
  "model.safetensors"
}

SAFETENSORS_INDEX_NAME <- function() {
  "model.safetensors.index.json"
}

state_dict_safetensors <- function(identifier, revision) {
  # first try safetensors file
  weights_path <- try(
    hub_download(identifier, SAFETENSORS_NAME(), revision=revision),
    silent = TRUE
  )

  if (!inherits(weights_path, "try-error")) {
    return(safetensors::safe_load_file(weights_path))
  }

  # now try the index
  index_path <- try(
    hub_download(identifier, SAFETENSORS_INDEX_NAME(), revision=revision),
    silent = TRUE
  )

  if (inherits(index_path, "try-error")) {
    cli::cli_abort(gettext(
      "No safetensors files found.",
      domain = "R-minhub"))
  }

  index <- jsonlite::fromJSON(index_path)$weight_map %>%
    unlist() %>%
    unique()

  index <- unname(sapply(index, function(fname) {
    hub_download(identifier, fname, revision = revision)
  }))

  do.call("c", lapply(index, safetensors::safe_load_file))
}
