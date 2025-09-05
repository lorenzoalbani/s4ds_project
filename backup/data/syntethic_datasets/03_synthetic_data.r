
library(MASS)
library(tidyverse)

# Replica Synthetic Dataset 3 (SD3) dal paper
# Case of Fair in Uncertainty, Unfair in Predictions

generate_synthetic_dataset_3 <- function(n_samples = 100) {
  set.seed(42)

# G=0,Y# Parameters per le 4 distribuzioni (dal paper)=0 ~ N([-2, -2], [[7,3],[3,7]])
# G=0,Y=0 ~ N([-2, -2], [[7,3],[3,7]])
# G=0,Y=1 ~ N([ 2,  2], [[7,3],[3,7]])
# G=1,Y=0 ~ N([-3, -3], [[5,3],[3,5]]) 
# G=1,Y=1 ~ N([ 3,  3], [[5,3],[3,5]])

  Sigma_g0 <- matrix(c(7, 3,
                       3, 7), nrow = 2, byrow = TRUE)
  Sigma_g1 <- matrix(c(5, 3,
                       3, 5), nrow = 2, byrow = TRUE)
  
  data_list <- list()
  
  # G=0, Y=0 - Normal distribution
  xy_00 <- mvrnorm(n_samples, mu = c(-2, -2), Sigma = Sigma_g0)
  data_list[[1]] <- data.frame(feature1 = xy_00[,1], feature2 = xy_00[,2], G = 0, Y = 0)
  
  # G=0, Y=1 - Normal distribution
  xy_01 <- mvrnorm(n_samples, mu = c( 2,  2), Sigma = Sigma_g0)
  data_list[[2]] <- data.frame(feature1 = xy_01[,1], feature2 = xy_01[,2], G = 0, Y = 1)
  
  # G=1, Y=0 - Normal distribution
  xy_10 <- mvrnorm(n_samples, mu = c(-3, -3), Sigma = Sigma_g1)
  data_list[[3]] <- data.frame(feature1 = xy_10[,1], feature2 = xy_10[,2], G = 1, Y = 0)
  
  # G=1, Y=1 - Normal distribution
  xy_11 <- mvrnorm(n_samples, mu = c( 3,  3), Sigma = Sigma_g1)
  data_list[[4]] <- data.frame(feature1 = xy_11[,1], feature2 = xy_11[,2], G = 1, Y = 1)
  
  # Combine all data
  synthetic_data <- do.call(rbind, data_list)
  synthetic_data$G <- as.factor(synthetic_data$G)
  synthetic_data$Y <- as.factor(synthetic_data$Y)
  
  return(synthetic_data)
}

# Test the function
sd3_data <- generate_synthetic_dataset_3()
head(sd3_data)
table(sd3_data$G, sd3_data$Y)

# --- Visualizzazione identica al tuo helper per SD1 ---
plot_synthetic_data <- function(data, title = "Synthetic Dataset") {
  ggplot(data, aes(x = feature1, y = feature2, 
                   color = interaction(G, Y), 
                   shape = interaction(G, Y))) +
    geom_point(size = 2, alpha = 0.7) +
    scale_color_manual(values = c("red", "blue", "green", "orange"),
                       labels = c("G=0,Y=0", "G=1,Y=0", "G=1,Y=1", "G=0,Y=1")) +
    scale_shape_manual(values = c(16, 17, 15, 18),
                       labels = c("G=0,Y=0", "G=1,Y=0", "G=1,Y=1", "G=0,Y=1")) +
    labs(title = title, 
         x = "Feature 1", 
         y = "Feature 2",
         color = "Group-Label",
         shape = "Group-Label") +
    theme_minimal()
}

p3 <- plot_synthetic_data(sd3_data, "Synthetic Dataset 3 - Fair in Uncertainty, Unfair in Predictions")
print(p3)
ggsave("plots/synthetic_dataset_3.png", p3, width = 10, height = 6)

