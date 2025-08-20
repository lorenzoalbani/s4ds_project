library(randomForest)
library(dplyr)

# Funzione per creare ensemble che simula Monte Carlo sampling
create_uncertainty_ensemble <- function(data, target_col, n_models = 10) {
  models <- list()
  
  # Crea formula dinamicamente
  formula_str <- paste(target_col, "~ .")
  
  for(i in 1:n_models) {
    # Bootstrap sampling (per aleatoric)
    boot_indices <- sample(nrow(data), nrow(data), replace = TRUE)
    boot_data <- data[boot_indices, ]
    
    # Aggiungi noise ai parametri (per epistemic) 
    set.seed(i)  # diversi seed per diversi modelli
    
    # Allena modello (esempio con randomForest)
    model <- randomForest(as.formula(formula_str), data = boot_data, 
                          mtry = sample(1:(ncol(data)-2), 1),  # varia parametri
                          ntree = sample(100:500, 1))
    models[[i]] <- model
  }
  return(models)
}