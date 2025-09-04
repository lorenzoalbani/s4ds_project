library(MASS)
library(tidyverse)

# Replica Synthetic Dataset 2 (SD2) dal paper
# Case of Epistemic Uncertainty

generate_synthetic_dataset_2 <- function(n_samples = 100) {
  set.seed(42)

  # Parameters per le 4 distribuzioni (dal paper)
  # G=0,Y=0 ~ N([-10,-10], [[100,30],[30,100]])
  # G=0,Y=1 ~ N([ 10, 10], [[100,30],[30,100]])
  # G=1,Y=0 ~ N([ -7, -7], [[  5, 1],[ 1,  5]])  # reso simmetrico
  # G=1,Y=1 ~ N([  7,  7], [[  5, 1],[ 1,  5]])

  Sigma_wide  <- matrix(c(100, 30,
                           30, 100), nrow = 2, byrow = TRUE)
  Sigma_tight <- matrix(c(  5,  1,
                             1,  5), nrow = 2, byrow = TRUE)
  
  data_list <- list()
  
  # G=0, Y=0 - Normal distribution
  n <- n_samples
  xy_00 <- mvrnorm(n, mu = c(-10, -10),
                    Sigma = Sigma_wide)
  data_list[[1]] <- data.frame(
    feature1 = xy_00[,1], 
    feature2 = xy_00[,2],
    G = 0, Y = 0
  )
  
  # G=0, Y=1 - Normal distribution
  xy_01 <- mvrnorm(n, mu = c(10, 10), 
                    Sigma = Sigma_wide)
  data_list[[2]] <- data.frame(
    feature1 = xy_01[,1], 
    feature2 = xy_01[,2], 
    G = 0, Y = 1
  )
  
  # G=1, Y=0 - Normal distribution
  xy_10 <- mvrnorm(n, mu = c(-7, -7), 
                    Sigma = Sigma_tight)
  data_list[[3]] <- data.frame(
    feature1 = xy_10[,1], 
    feature2 = xy_10[,2], 
    G = 1, Y = 0
  )
  
  # G=1, Y=1 - Normal distribution
  xy_11 <- mvrnorm(n, mu = c(7, 7), 
                    Sigma = Sigma_tight)
  data_list[[4]] <- data.frame(
    feature1 = xy_11[,1], 
    feature2 = xy_11[,2], 
    G = 1, Y = 1
  )
  
  # Combine all data
  synthetic_data <- do.call(rbind, data_list)
  synthetic_data$G <- as.factor(synthetic_data$G)
  synthetic_data$Y <- as.factor(synthetic_data$Y)
  
  return(synthetic_data)
}

# Test the function
sd2_data <- generate_synthetic_dataset_2()
head(sd2_data)
table(sd2_data$G, sd2_data$Y)

# --- Visualizzazione identica al tuo helper per SD1 ---
plot_synthetic_data <- function(data, title = "Synthetic Dataset") {
  ggplot(data, aes(x = feature1, y = feature2, 
                   color = interaction(G, Y), 
                   shape = interaction(G, Y))) +
    geom_point(size = 2, alpha = 0.7) +
    scale_color_manual(values = c("red", "blue", "green", "orange"),
                       labels = c("G=0,Y=0", "G=1,Y=0", "G=0,Y=1", "G=1,Y=1")) +
    scale_shape_manual(values = c(16, 17, 15, 18),
                       labels = c("G=0,Y=0", "G=1,Y=0", "G=0,Y=1", "G=1,Y=1")) +
    labs(title = title, 
         x = "Feature 1", 
         y = "Feature 2",
         color = "Group-Label",
         shape = "Group-Label") +
    theme_minimal()
}

p2 <- plot_synthetic_data(sd2_data, "Synthetic Dataset 2 - Epistemic Uncertainty Case")
print(p2)
ggsave("plots/synthetic_dataset_2.png", p2, width = 10, height = 6)

