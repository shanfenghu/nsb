# This script performs the following steps:
# 1. Loads both the raw R data file (`.rds`) and the processed CSV.
# 2. Calculates key summary statistics, including the number of outbreaks,
#    individuals, transmissions, and distributional properties.
# 3. Saves these statistics to a clean CSV file.
# 4. Generates a publication-quality LaTeX table of the statistics.

# --- Load necessary libraries ---
library(tidyverse)
library(here)    # For robust path management
library(igraph)  # For processing the raw graph objects
library(xtable)  # For generating LaTeX tables

# --- Configuration ---
# 'here()' automatically finds the project root directory
RAW_DATA_PATH <- here("data", "taube_2020_transmission_trees.rds")
PROCESSED_DATA_PATH <- here("data", "outbreaktrees_sars_mers_counts.csv")
OUTPUT_CSV_PATH <- here("results", "outbreak_data_summary.csv")
OUTPUT_TEX_PATH <- here("results", "outbreak_data_summary.tex")

# --- Main Summarization Logic ---
cat("--- Starting Data Summarization ---\n")

# --- Step 1: Load Raw and Processed Data ---
cat("Loading data...\n")
outbreak_data <- readRDS(RAW_DATA_PATH)
df_processed <- read_csv(PROCESSED_DATA_PATH, col_types = cols(offspring_count = col_integer()))

# --- Step 2: Calculate Statistics from Raw Data ---
diseases_to_include <- c("SARS", "MERS")
df_filtered <- outbreak_data %>%
  filter(Disease %in% diseases_to_include)

num_outbreaks <- nrow(df_filtered)
all_edges <- list()
all_individuals <- c()

for (i in 1:nrow(df_filtered)) {
  row <- df_filtered[i, ]
  tree_mod_object <- row$tree_mod[[1]] 
  
  if (inherits(tree_mod_object, "igraph")) {
    edges <- igraph::as_data_frame(tree_mod_object, what = "edges")
    names(edges) <- c("infector", "case_id")
    outbreak_prefix <- paste0(row$id, "_")
    edges$infector <- paste0(outbreak_prefix, edges$infector)
    edges$case_id <- paste0(outbreak_prefix, edges$case_id)
    all_edges[[i]] <- edges
    all_individuals <- c(all_individuals, edges$infector, edges$case_id)
  }
}
df_edges <- bind_rows(all_edges)
total_individuals <- length(unique(all_individuals))
total_transmissions <- nrow(df_edges)

# --- Step 3: Calculate Statistics from Processed Data ---
mean_offspring <- mean(df_processed$offspring_count)
var_offspring <- var(df_processed$offspring_count)
max_offspring <- max(df_processed$offspring_count)
percent_zeros <- (sum(df_processed$offspring_count == 0) / nrow(df_processed)) * 100

# --- Step 4: Assemble the Summary Table ---
summary_df <- tibble(
  Statistic = c(
    "Number of Outbreaks (SARS & MERS)",
    "Total Individuals (Nodes)",
    "Total Transmissions (Edges)",
    "Mean Offspring Count",
    "Variance of Offspring Count",
    "Maximum Offspring Count",
    "Percentage of Zeros"
  ),
  Value = c(
    num_outbreaks,
    total_individuals,
    total_transmissions,
    round(mean_offspring, 3),
    round(var_offspring, 1),
    max_offspring,
    paste0(round(percent_zeros, 1), "%")
  )
)

cat("Generated Summary:\n")
print(summary_df)

# --- Step 5: Save Outputs ---
# Save as CSV
write_csv(summary_df, OUTPUT_CSV_PATH)
cat(paste("\nSummary table saved as CSV to '", OUTPUT_CSV_PATH, "'\n", sep=""))

# Generate and save as LaTeX table
latex_table <- xtable(
  summary_df,
  caption = "Summary statistics for the processed `outbreaktrees` dataset (SARS and MERS).",
  label = "tab:data_summary"
)
print(latex_table, file = OUTPUT_TEX_PATH, caption.placement = "top", include.rownames = FALSE)
cat(paste("Summary table saved as LaTeX to '", OUTPUT_TEX_PATH, "'\n", sep=""))

cat("\n--- Summarization Complete ---\n")
