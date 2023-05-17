#' Loads, and possibly downloads HF Hub models
#' @param identifier Repository id from the model
#' @param revision Revision to download from (eg tags, branches or commit hashes)
#' @importFrom hfhub hub_download WEIGHTS_NAME WEIGHTS_INDEX_NAME
#' @export
hf_state_dict <- function(identifier, revision = "main") {
  weights_path <- try(hub_download(identifier, WEIGHTS_NAME(), revision = revision), silent = TRUE)
  if (inherits(weights_path, "try-error")) {
    index_path <- hub_download(identifier, WEIGHTS_INDEX_NAME(), revision = revision)
    filenames <- unique(unlist(jsonlite::fromJSON(index_path)$weight_map))
    weights_path <- sapply(filenames, function(fname) {
      hub_download(identifier, fname, revision = revision)
    })
    names(weights_path) <- NULL
  }
  do.call("c", lapply(weights_path, torch::load_state_dict))
}

