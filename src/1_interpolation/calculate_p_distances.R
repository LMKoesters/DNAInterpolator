options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressMessages(if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman"))
pacman::p_load("optparse")
pacman::p_load("seqinr")
pacman::p_load("bit64")
pacman::p_load("ape")
pacman::p_load("tidyr")

option_list <- list(
  make_option(c("-a", "--alignment"),
    type = "character", default = NULL,
    help = "alignment file path", metavar = "character"
  ),
  make_option(c("-o", "--out"),
    type = "character", default = "out.txt",
    help = "output file path [default= %default]", metavar = "character"
  ),
  make_option(c("-t", "--type"),
    type = "character", default = "fasta",
    help = "alignment file type [default= %default]", metavar = "character"
  )
)
#' This script reads a given alignments and uses as.DNAbin and dist.dna
#' to produce a pairwise distance matrix based on the proportion of
#' differing sites to the complete sequence length

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

aln <- read.alignment(opt$alignment, opt$type)
aln_bin <- as.DNAbin(aln)
dist_matrix <- dist.dna(aln_bin, model = "raw", as.matrix = TRUE)

dist_df <- as.data.frame(dist_matrix)
dist_df["sampleA"] <- rownames(dist_df)
rownames(dist_df) <- seq_len(nrow(dist_df))
dist_df <- dist_df |>
  pivot_longer(
    cols = -c("sampleA"),
    names_to = "sampleB",
    values_to = "p_distance"
  )

write.table(
  dist_df,
  file = opt$out,
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)
