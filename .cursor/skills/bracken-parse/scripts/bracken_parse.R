#!/usr/bin/env Rscript
# Skill entry → shared bracken_parse implementation
root <- local({
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) {
    script <- normalizePath(sub("^--file=", "", f[[1]]), mustWork = FALSE)
    return(dirname(dirname(dirname(dirname(dirname(script))))))
  }
  getwd()
})
status <- system2(
  "Rscript",
  c(file.path(root, ".cursor/skills/_shared/import/bracken_parse.R"), commandArgs(trailingOnly = TRUE))
)
quit(save = "no", status = if (is.null(status)) 0L else as.integer(status))
