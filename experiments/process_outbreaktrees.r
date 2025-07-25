# Pre-processes the raw `outbreaktrees` dataset using R.
#
# This script performs the following steps:
# 1. Loads the raw R data file (`.rds`).
# 2. Filters the data to include only SARS and MERS outbreaks.
# 3. Parses the 'igraph' objects from the 'tree_mod' column to extract edges.
# 4. Calculates the "offspring count" for each individual.
# 5. Correctly accounts for individuals who infected zero people.
# 6. Saves the final, clean, single-column dataset to a CSV file.

# --- Load necessary libraries ---
# If you don't have these, run: install.packages(c("tidyverse", "here", "igraph"))
library(tidyverse)
library(here)
library(igraph)

# --- Configuration ---
RAW_DATA_PATH <- here("data", "taube_2020_transmission_trees.rds")
PROCESSED_DATA_PATH <- here("data", "outbreaktrees_sars_mers_counts.csv")

# --- Main Processing Logic ---
cat("--- Starting data processing for OutbreakTrees dataset ---\n")

# --- Step 1: Load Raw Data ---
cat(paste("Loading raw data from '", RAW_DATA_PATH, "'...\n", sep=""))
if (!file.exists(RAW_DATA_PATH)) {
  stop("Raw data file not found. Please download and place it in the 'data/' directory.")
}
outbreak_data <- readRDS(RAW_DATA_PATH)
cat(paste("Successfully loaded", nrow(outbreak_data), "total outbreak records.\n"))

# --- Step 2: Filter for Relevant Diseases ---
diseases_to_include <- c("SARS", "MERS")
df_filtered <- outbreak_data %>%
  filter(Disease %in% diseases_to_include)
cat(paste("Filtered down to", nrow(df_filtered), "records for", paste(diseases_to_include, collapse=", "), ".\n"))

# --- Step 3: Extract All Transmission Edges ---
all_edges <- list()
all_individuals <- c()

cat("Processing transmission trees for each outbreak...\n")
for (i in 1:nrow(df_filtered)) {
  row <- df_filtered[i, ]
  tree_mod_object <- row$tree_mod[[1]] 
  
  # Use the igraph library to correctly extract edges ---
  if (inherits(tree_mod_object, "igraph")) {
    # as_data_frame is the correct way to extract edges from an igraph object
    edges <- igraph::as_data_frame(tree_mod_object, what = "edges")
    
    # The columns are named 'from' and 'to'. We rename them.
    names(edges) <- c("infector", "case_id")
    
    # Create unique IDs to avoid collisions between outbreaks
    outbreak_prefix <- paste0(row$id, "_")
    edges$infector <- paste0(outbreak_prefix, edges$infector)
    edges$case_id <- paste0(outbreak_prefix, edges$case_id)
    
    all_edges[[i]] <- edges
    all_individuals <- c(all_individuals, edges$infector, edges$case_id)
  }
}

df_edges <- bind_rows(all_edges)
cat(paste("\nSuccessfully extracted", nrow(df_edges), "total transmission edges.\n"))

# --- Step 4: Calculate Offspring Counts and Handle Zeros ---
if(nrow(df_edges) > 0) {
  offspring_counts <- df_edges %>%
    count(infector, name = "offspring_count") %>%
    rename(individual_id = infector)
  cat(paste("Found", nrow(offspring_counts), "individuals who infected at least one other person.\n"))

  all_individuals_df <- tibble(individual_id = unique(all_individuals))
  cat(paste("Total unique individuals across SARS/MERS outbreaks:", nrow(all_individuals_df), "\n"))

  df_final <- all_individuals_df %>%
    left_join(offspring_counts, by = "individual_id") %>%
    # Replace NA with 0 for individuals who infected no one
    mutate(offspring_count = replace_na(offspring_count, 0))

  # --- Step 5: Finalize and Save ---
  df_processed <- df_final %>%
    select(offspring_count)

  write_csv(df_processed, PROCESSED_DATA_PATH)

  cat("\n--- Data Processing Complete ---\n")
  cat(paste("Processed data saved to '", PROCESSED_DATA_PATH, "'\n", sep=""))
  cat("Total number of individuals (data points):", nrow(df_processed), "\n")
  cat("Offspring count summary:\n")
  print(summary(df_processed$offspring_count))
} else {
  cat("\n--- Data Processing Halted: No edges were extracted. ---\n")
}
