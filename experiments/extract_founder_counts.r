# Extracts founder counts (z) and cluster sizes (n) for each outbreak.
#
# This script performs the following steps:
# 1. Loads the raw R data file (`.rds`).
# 2. Filters the data to include only SARS and MERS outbreaks.
# 3. For each outbreak, extracts the transmission tree (igraph object).
# 4. Identifies founders (nodes with no incoming edges).
# 5. Counts the number of founders (z) and total cluster size (n) per outbreak.
# 6. Saves the results to a CSV file with columns: outbreak_id, cluster_size_n, founder_count_z

# --- Load necessary libraries ---
# If you don't have these, run: install.packages(c("dplyr", "readr", "here", "igraph"))
library(dplyr)
library(readr)
library(here)
library(igraph)

# --- Configuration ---
RAW_DATA_PATH <- here("data", "taube_2020_transmission_trees.rds")
OUTPUT_CSV_PATH <- here("data", "outbreaktrees_founder_counts.csv")

# --- Main Extraction Logic ---
cat("--- Starting Founder Count Extraction ---\n")

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

# --- Step 3: Extract Founder Counts and Cluster Sizes ---
cat("Processing transmission trees to extract founder counts...\n")

outbreak_results <- list()

for (i in 1:nrow(df_filtered)) {
  row <- df_filtered[i, ]
  outbreak_id <- row$id
  tree_mod_object <- row$tree_mod[[1]]
  
  if (inherits(tree_mod_object, "igraph")) {
    # Get all nodes in the graph
    all_nodes <- V(tree_mod_object)
    cluster_size_n <- length(all_nodes)
    
    # Find founders: nodes with no incoming edges (in-degree = 0)
    in_degrees <- degree(tree_mod_object, mode = "in")
    founders <- which(in_degrees == 0)
    founder_count_z <- length(founders)
    
    # Store results
    outbreak_results[[i]] <- tibble(
      outbreak_id = outbreak_id,
      cluster_size_n = cluster_size_n,
      founder_count_z = founder_count_z
    )
    
    if (i %% 10 == 0) {
      cat(paste("Processed", i, "outbreaks...\n"))
    }
  } else {
    # If not an igraph object, skip with warning
    cat(paste("Warning: Outbreak", outbreak_id, "does not have a valid igraph object. Skipping.\n"))
  }
}

# --- Step 4: Combine Results and Save ---
df_founder_counts <- bind_rows(outbreak_results)

cat("\n--- Extraction Complete ---\n")
cat(paste("Successfully processed", nrow(df_founder_counts), "outbreaks.\n"))
cat("\nSummary of founder counts:\n")
print(summary(df_founder_counts$founder_count_z))
cat("\nSummary of cluster sizes:\n")
print(summary(df_founder_counts$cluster_size_n))

# Save to CSV
write_csv(df_founder_counts, OUTPUT_CSV_PATH)
cat(paste("\nFounder count data saved to '", OUTPUT_CSV_PATH, "'\n", sep=""))

cat("\n--- Extraction Complete ---\n")

