# 3. FUNZIONE CORRETTA PER CALCOLARE ACCURACY CON RANDOM FOREST
calculate_rf_ensemble_accuracy <- function(models, test_data, target_col) {
  
  # Vettore per memorizzare le accuracy individuali
  individual_accuracies <- numeric(length(models))
  
  # Lista per memorizzare tutte le predizioni
  all_predictions <- list()
  
  # Calcola predizioni per ogni modello
  for(i in 1:length(models)) {
    
    # Predizioni del modello i-esimo (Random Forest restituisce fattori)
    predictions <- predict(models[[i]], newdata = test_data)
    
    # Calcola accuracy per questo modello
    actual_values <- test_data[[target_col]]
    accuracy_i <- mean(predictions == actual_values)
    individual_accuracies[i] <- accuracy_i
    
    # Salva le predizioni (converti in numerico per ensemble)
    all_predictions[[i]] <- as.numeric(as.character(predictions))
    
    cat("Modello", i, "- Accuracy:", round(accuracy_i, 4), "\n")
  }
  
  # Calcola accuracy media dei modelli individuali
  mean_individual_accuracy <- mean(individual_accuracies)
  
  # Calcola ensemble prediction (moda delle predizioni)
  # Per classificazione, usiamo la moda invece della media
  ensemble_predictions <- apply(do.call(cbind, all_predictions), 1, function(x) {
    # Trova la moda (valore più frequente)
    tbl <- table(x)
    as.numeric(names(tbl)[which.max(tbl)])
  })
  
  # Converti ensemble predictions in fattore
  ensemble_predictions <- factor(ensemble_predictions, levels = levels(test_data[[target_col]]))
  
  # Calcola accuracy dell'ensemble
  ensemble_accuracy <- mean(ensemble_predictions == test_data[[target_col]])
  
  # Restituisci risultati
  results <- list(
    individual_accuracies = individual_accuracies,
    mean_individual_accuracy = mean_individual_accuracy,
    ensemble_accuracy = ensemble_accuracy,
    ensemble_predictions = ensemble_predictions
  )
  
  return(results)
}

# Funzione per calcolare accuracy per gruppi con Random Forest
calculate_rf_group_accuracy <- function(models, test_data, sensitive_attr, target_col = "y") {
  groups <- unique(test_data[[sensitive_attr]])
  
  results <- data.frame(
    Group = character(),
    Accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (group in groups) {
    group_data <- test_data[test_data[[sensitive_attr]] == group, ]
    true_labels <- group_data[[target_col]]
    
    # Calcola accuracy per ogni modello su questo gruppo
    accuracies <- sapply(models, function(model) {
      predictions <- predict(model, newdata = group_data)
      mean(predictions == true_labels)
    })
    
    # Accuracy media per questo gruppo
    group_accuracy <- mean(accuracies)
    
    results <- rbind(results, data.frame(
      Group = as.character(group),
      Accuracy = group_accuracy
    ))
    
    cat("Gruppo", group, "- Accuracy:", round(group_accuracy, 4), "\n")
  }
  
  return(results)
}
