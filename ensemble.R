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

calculate_group_accuracy <- function(models, test_data, sensitive_attr) {
  groups <- unique(test_data[[sensitive_attr]])
  
  results <- data.frame(
    Group = character(),
    Accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (group in groups) {
    group_data <- test_data[test_data[[sensitive_attr]] == group, ]
    
    accuracies <- sapply(models, function(model) {
      preds <- predict(model, group_data)
      mean(preds == group_data$Y)
    })
    
    results <- rbind(results, data.frame(
      Group = as.character(group),
      Accuracy = mean(accuracies)
    ))
  }
  
  return(results)
}


calculate_classification_metrics <- function(models, test_data, target_col = "Y", threshold = 0.5) {
  predictions_list <- lapply(models, function(model) {
    predict(model, test_data, type = "response")
  })
  
  avg_predictions <- rowMeans(do.call(cbind, predictions_list))
  predicted_classes <- ifelse(avg_predictions > threshold, 1, 0)
  true_classes <- test_data[[target_col]]
  
  # Assicura che la tabella abbia tutti i livelli
  conf_matrix <- table(factor(predicted_classes, levels = c(0, 1)),
                       factor(true_classes, levels = c(0, 1)))
  
  TP <- conf_matrix["1", "1"]
  TN <- conf_matrix["0", "0"]
  FP <- conf_matrix["1", "0"]
  FN <- conf_matrix["0", "1"]
  
  accuracy <- (TP + TN) / sum(conf_matrix)
  TPR <- if ((TP + FN) > 0) TP / (TP + FN) else NA
  TNR <- if ((TN + FP) > 0) TN / (TN + FP) else NA
  
  return(list(
    Accuracy = accuracy,
    True_Positive_Rate = TPR,
    True_Negative_Rate = TNR,
    False_Positive = FP,
    False_Negative = FN
  ))
}

