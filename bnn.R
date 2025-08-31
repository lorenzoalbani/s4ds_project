library(bnns)  # Bayesian Neural Networks
library(dplyr)

# Funzione per creare ensemble che simula Monte Carlo sampling
create_uncertainty_ensemble <- function(data, target_col, n_models = 2) {
  models <- list()
  
  # Crea formula dinamicamente
  formula_str <- paste(target_col, "~ .")
  
  for(i in 1:n_models) {
    cat("Training model", i, "of", n_models, "\n")
    # Bootstrap sampling (per aleatoric)
    boot_indices <- sample(nrow(data), nrow(data), replace = TRUE)
    boot_data <- data[boot_indices, ]
    
    # Aggiungi noise ai parametri (per epistemic) 
    set.seed(i)  # diversi seed per diversi modelli
    
    # SOSTITUITO: Allena BNN con bnns
    model <- bnns(
      formula = as.formula(formula_str),
      data = boot_data,
      hidden_layers = 10,  # varia architettura
      epochs = 50,               # varia training
      seed = i
    )
    
    models[[i]] <- model
  }
  return(models)
}


calculate_group_accuracy <- function(models, test_data, sensitive_attr, train_data, target_col = "y") {
  groups <- unique(test_data[[sensitive_attr]])
  feature_names <- setdiff(names(train_data), c(target_col, sensitive_attr))
  
  results <- data.frame(
    Group = character(),
    Accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (group in groups) {
    group_data <- test_data[test_data[[sensitive_attr]] == group, ]
    true_labels <- group_data[[target_col]]
    group_data_x <- group_data[, feature_names]
    group_data_x <- as.data.frame(lapply(group_data_x, as.numeric))
    
    accuracies <- sapply(models, function(model) {
      # 10 campioni Monte Carlo dalla posteriore bayesiana
      mc_preds <- replicate(2, predict(model, group_data, type = "response"))
      avg_pred <- rowMeans(mc_preds)
      predicted_classes <- ifelse(avg_pred > 0.5, 1, 0)
      mean(predicted_classes == true_labels)
    })
    
    results <- rbind(results, data.frame(
      Group = as.character(group),
      Accuracy = mean(accuracies)
    ))
  }
  
  return(results)
}


prepare_test_data <- function(train_data, test_data, target_col, sensitive_col) {
  feature_names <- setdiff(names(train_data), c(target_col, sensitive_col))
  
  # Seleziona solo le feature
  test_x <- test_data[, feature_names]
  
  # Assicurati che siano numeriche
  test_x <- as.data.frame(lapply(test_x, as.numeric))
  
  return(test_x)
}





calculate_classification_metrics <- function(models, test_data, target_col = "Y", threshold = 0.5) {
  # MODIFICATO: Monte Carlo sampling con BNN
  predictions_list <- lapply(models, function(model) {
    mc_preds <- replicate(2, predict(model, test_data, type = "response"))
    rowMeans(mc_preds)  # media dei 10 campioni MC
  })
  
  avg_predictions <- rowMeans(do.call(cbind, predictions_list))
  predicted_classes <- ifelse(avg_predictions > threshold, 1, 0)
  true_classes <- test_data[[target_col]]
  
  # Resto identico...
  conf_matrix <- table(factor(predicted_classes, levels = c(0, 1)),
                       factor(true_classes, levels = c(0, 1)))
  
  TP <- conf_matrix["1", "1"]
  TN <- conf_matrix["0", "0"]
  FP <- conf_matrix["1", "0"]
  FN <- conf_matrix["0", "1"]
  
  accuracy <- (TP + TN) / sum(conf_matrix)
  TPR <- if ((TP + FN) > 0) TP / (TP + FN) else NA
  TNR <- if ((TN + FP) > 0) TN / (TN + FP) else NA
  PPV <- if ((TP + FP) > 0) TP / (TP + FP) else NA
  NPV <- if ((TN + FN) > 0) TN / (TN + FN) else NA
  FPR <- if ((FP + TN) > 0) FP / (FP + TN) else NA
  FNR <- if ((FN + TP) > 0) FN / (FN + TP) else NA
  
  return(list(
    Accuracy = accuracy,
    Positive_Predictive_Value = PPV,
    Negative_Predictive_Value = NPV,
    True_Positive_Rate = TPR,
    True_Negative_Rate = TNR,
    False_Positive_Rate = FPR,
    False_Negative_Rate = FNR
  ))
}

# Calcola uncertainties seguendo Eq. 7 del paper
calculate_uncertainties <- function(ensemble_models, test_data) {
  M <- length(ensemble_models)
  n_samples <- nrow(test_data)
  
  # MODIFICATO: Monte Carlo per ogni modello BNN
  P_matrix <- matrix(0, nrow = n_samples, ncol = M)
  
  # MODIFICATO: Monte Carlo sampling con BNN  
  for(m in 1:M) {
    # 10 campioni Monte Carlo dalla posteriore bayesiana
    mc_preds <- replicate(2, predict(ensemble_models[[m]], test_data, type = "response"))
    P_matrix[,m] <- rowMeans(mc_preds)  # media dei campioni MC
  }
  
  # Resto identico al tuo codice originale...
  P_bar <- rowMeans(P_matrix)
  
  # EPISTEMIC UNCERTAINTY (Eq. 7 prima parte)
  epistemic <- numeric(n_samples)
  for(i in 1:n_samples) {
    diff <- P_matrix[i,] - P_bar[i]
    epistemic[i] <- mean(diff^2)
  }
  
  # ALEATORIC UNCERTAINTY (Eq. 7 seconda parte)  
  aleatoric <- numeric(n_samples)
  for(i in 1:n_samples) {
    for(m in 1:M) {
      p_m <- P_matrix[i,m]
      aleatoric[i] <- aleatoric[i] + (p_m - p_m^2)
    }
    aleatoric[i] <- aleatoric[i] / M
  }
  
  # PREDICTIVE UNCERTAINTY = Epistemic + Aleatoric
  predictive <- epistemic + aleatoric
  
  return(list(
    epistemic = epistemic,
    aleatoric = aleatoric, 
    predictive = predictive
  ))
}