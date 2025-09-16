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
    # Versione degradata per far emergere le uncertainties
    model <- randomForest(as.formula(formula_str), data = boot_data, 
                          ntree = 50,        # pochissimi alberi
                          mtry = 2,         # solo 1 feature 
                          nodesize = 3,    # nodi molto grandi
                          maxnodes = 100)     # alberi piccolissimi
    models[[i]] <- model
  }
  return(models)
}


calculate_group_metrics <- function(models, test_data, sensitive_attr) {
  groups <- unique(test_data[[sensitive_attr]])
  
  results <- data.frame(
    Group = character(),
    Accuracy = numeric(),
    TPR = numeric(),
    TNR = numeric(),
    FPR = numeric(),
    FNR = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (group in groups) {
    group_data <- test_data[test_data[[sensitive_attr]] == group, ]
    
    # Media delle predizioni dei modelli (voting ensemble)
    preds_matrix <- sapply(models, function(model) {
      predict(model, group_data)
    })
    
    # Majority vote
    final_preds <- apply(preds_matrix, 1, function(row) {
      names(sort(table(row), decreasing = TRUE))[1]
    })
    
    # Confusion matrix components
    actuals <- group_data$Y
    TP <- sum(final_preds == 1 & actuals == 1)
    TN <- sum(final_preds == 0 & actuals == 0)
    FP <- sum(final_preds == 1 & actuals == 0)
    FN <- sum(final_preds == 0 & actuals == 1)
    
    # Metriche
    accuracy <- mean(final_preds == actuals)
    tpr <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    tnr <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
    fpr <- ifelse((FP + TN) > 0, FP / (FP + TN), NA)
    fnr <- ifelse((FN + TP) > 0, FN / (FN + TP), NA)
    
    results <- rbind(results, data.frame(
      Group = as.character(group),
      Accuracy = round(accuracy, 2),
      TPR = round(tpr, 2),
      TNR = round(tnr, 2),
      FPR = round(fpr, 2),
      FNR = round(fnr, 2)
    ))
  }
  
  return(results)
}


