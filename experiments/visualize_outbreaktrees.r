# Visualizes a single, large transmission tree from the `outbreaktrees` dataset.
#
# This script performs the following steps:
# 1. Loads the raw R data file (`.rds`).
# 2. Selects a specific, large SARS outbreak to visualize.
# 3. Extracts the `igraph` object for this outbreak.
# 4. Calculates the out-degree (offspring count) for each node.
# 5. Uses the `ggraph` library to create a high-quality plot of the
#    transmission tree, highlighting superspreaders by node size and color.
# 6. Saves the final plot to a PDF file.

# --- Load necessary libraries ---
# If you don't have these, run: install.packages(c("tidyverse", "here", "igraph", "ggraph", "Cairo"))
library(tidyverse)
library(here)    # For robust path management
library(igraph)  # For graph manipulation
library(ggraph)  # For beautiful graph plotting
library(Cairo)   # For robust PDF generation

# --- Configuration ---
# 'here()' automatically finds the project root directory
RAW_DATA_PATH <- here("data", "taube_2020_transmission_trees.rds")
OUTPUT_FIGURE_PATH <- here("figures", "sars_transmission_tree.pdf")

# The ID of the specific outbreak we want to visualize.
# 'chn.2003.sars.1.00' is a large, well-documented SARS outbreak.
TARGET_OUTBREAK_ID <- "chn.2003.sars.1.00"

# --- Main Visualization Logic ---
cat("--- Starting Outbreak Visualization ---\n")

# --- Step 1: Load Raw Data ---
cat(paste("Loading raw data from '", RAW_DATA_PATH, "'...\n", sep=""))
if (!file.exists(RAW_DATA_PATH)) {
  stop("Raw data file not found. Please download and place it in the 'data/' directory.")
}
outbreak_data <- readRDS(RAW_DATA_PATH)
cat("Successfully loaded outbreak records.\n")

# --- Step 2: Select Target Outbreak and Extract Graph ---
target_outbreak <- outbreak_data %>%
  filter(id == TARGET_OUTBREAK_ID)

if (nrow(target_outbreak) == 0) {
  stop(paste("Could not find the target outbreak with ID:", TARGET_OUTBREAK_ID))
}

# The graph object is nested in the 'tree_mod' column
transmission_graph <- target_outbreak$tree_mod[[1]]
cat(paste("Successfully extracted the '", TARGET_OUTBREAK_ID, "' transmission graph.\n", sep=""))

# --- Step 3: Calculate Out-Degree (Offspring Count) ---
# The out-degree of a node is the number of other nodes it points to,
# which is exactly the number of secondary infections.
out_degrees <- degree(transmission_graph, mode = "out")

# Add the out-degree as an attribute to the graph's vertices (nodes)
V(transmission_graph)$offspring_count <- out_degrees

# --- Step 4: Generate the Plot with ggraph ---
cat("Generating the visualization...\n")

# Create the plot object
tree_plot <- ggraph(transmission_graph, layout = 'tree') + 
  # Draw the edges (transmission paths) as simple lines
  geom_edge_fan(arrow = arrow(length = unit(2, 'mm')), 
                end_cap = circle(3, 'mm')) +
  # Draw the nodes (individuals)
  geom_node_point(aes(size = offspring_count, color = offspring_count), alpha = 0.8) +
  # Use a color scale that highlights high values
  scale_color_gradient(low = "lightblue", high = "red") +
  # Use a size scale that makes superspreaders stand out
  scale_size_continuous(range = c(2, 10)) +
  # Add labels for the number of infections to the larger nodes
  geom_node_text(aes(label = ifelse(offspring_count > 4, offspring_count, "")), 
                 repel = TRUE, color = "black", size = 3) +
  # Apply a clean, minimal theme
  theme_graph() +
  # Add titles and legends
  labs(
    title = paste("Transmission Tree for SARS Outbreak:", TARGET_OUTBREAK_ID),
    subtitle = "Node size and color represent the number of secondary infections (offspring count)",
    size = "Offspring Count",
    color = "Offspring Count"
  )

# --- Step 5: Save the Figure ---
ggsave(
  OUTPUT_FIGURE_PATH,
  plot = tree_plot,
  width = 10,
  height = 8,
  units = "in",
  # --- FIX: Use the cairo_pdf device for better font handling ---
  device = cairo_pdf
)

cat(paste("\n--- Visualization Complete ---\n"))
cat(paste("Transmission tree plot saved to '", OUTPUT_FIGURE_PATH, "'\n", sep=""))

