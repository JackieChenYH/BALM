# Training adapter for the external tensorsparse 3.0 package.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 7L) stop("Expected: output_dir n L M K max_iter tolerance")
output <- args[1]
n <- as.integer(args[2]); L <- as.integer(args[3])
M <- as.integer(args[4]); K <- as.integer(args[5])
max_iter <- as.integer(args[6]); tolerance <- as.numeric(args[7])
if (!requireNamespace("tensorsparse", quietly=TRUE)) {
  stop("Install tensorsparse 3.0 in R: install.packages('tensorsparse', repos='https://cloud.r-project.org')")
}
if (as.character(utils::packageVersion("tensorsparse")) != "3.0") {
  stop("This reproduction requires tensorsparse version 3.0")
}
options(digits=17)
writeLines(capture.output(sessionInfo()), file.path(output, "R_sessionInfo.txt"))
P <- n * (n - 1L) / 2L
input <- read.csv(file.path(output, "training.csv"), na.strings="", check.names=FALSE)
stopifnot(nrow(input) == L * P,
          identical(input$subject, rep(0:(L-1L), each=P)),
          identical(input$edge, rep(0:(P-1L), times=L)))
mask <- matrix(input$train == 1L, nrow=L, byrow=TRUE)
z <- matrix(input$z, nrow=L, byrow=TRUE)
stopifnot(all(is.na(z[!mask])), all(!is.na(z[mask])), all(z[mask] %in% c(0, 1)))
global_mean <- mean(z[mask])
means <- colSums(replace(z, !mask, 0)) / pmax(colSums(mask), 1)
means[colSums(mask) == 0L] <- global_mean
filled <- z
for (edge in seq_len(P)) filled[!mask[, edge], edge] <- means[edge]
A <- array(0, dim=c(L, n, n))
node_i <- input$node_i[seq_len(P)] + 1L
node_j <- input$node_j[seq_len(P)] + 1L
for (edge in seq_len(P)) {
  A[, node_i[edge], node_j[edge]] <- filled[, edge]
  A[, node_j[edge], node_i[edge]] <- filled[, edge]
}
write.csv(data.frame(edge=0:(P-1L), training_mean=means), file.path(output, "imputation.csv"), row.names=FALSE)
seeds <- read.csv(file.path(output, "seeds.csv"))
records <- list(); best <- NULL; best_loss <- Inf; best_start <- NA_integer_
for (index in seq_len(nrow(seeds))) {
  set.seed(seeds$seed[index])
  messages <- character()
  fit <- withCallingHandlers(
    tensorsparse::tbmClustering(A, k=M, r=K, l=K, lambda=0, max.iter=max_iter,
                               threshold=tolerance, sim.times=1L, method="L0"),
    warning=function(w) {
      messages <<- c(messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  if (length(fit$objs) != 1L || !is.finite(fit$objs) || any(!is.finite(fit$judgeX))) {
    stop("TBM returned a nonfinite result")
  }
  saveRDS(list(fit=fit, seed=seeds$seed[index], warnings=messages),
          file.path(output, sprintf("start_%02d.rds", seeds$start[index])))
  records[[index]] <- data.frame(start=seeds$start[index], seed=seeds$seed[index],
                                training_filled_SSE=fit$objs,
                                iteration_cap_warning=any(grepl("not converged", messages, fixed=TRUE)),
                                actual_layer_groups=length(unique(fit$Cs)),
                                actual_node_i_groups=length(unique(fit$Ds)),
                                actual_node_j_groups=length(unique(fit$Es)))
  write.csv(do.call(rbind, records), file.path(output, "starts.csv"), row.names=FALSE)
  if (fit$objs < best_loss) {
    best <- fit; best_loss <- fit$objs; best_start <- seeds$start[index]
  }
}
scores <- matrix(NA_real_, L, P)
for (edge in seq_len(P)) {
  scores[, edge] <- (best$judgeX[, node_i[edge], node_j[edge]] +
                    best$judgeX[, node_j[edge], node_i[edge]]) / 2
}
write.csv(data.frame(subject=rep(0:(L-1L), each=P), edge=rep(0:(P-1L), times=L),
                     topology_score=as.vector(t(scores))), file.path(output, "topology_scores.csv"), row.names=FALSE)
write.csv(data.frame(subject=0:(L-1L), label=best$Cs-1L), file.path(output, "labels.csv"), row.names=FALSE)
write.csv(data.frame(selected_start=best_start, training_filled_SSE=best_loss),
          file.path(output, "selected.csv"), row.names=FALSE)
