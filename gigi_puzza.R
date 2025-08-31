# Funzione semplice per calcolare accuracy media
calculate_mean_accuracy <- function(models, test_data, target_col) {
  
  accuracies <- numeric(length(models))
  
  for(i in 1:length(models)) {
    # Predizioni del modello i-esimo
    predictions <- predict(models[[i]], newdata = test_data)
    
    # Converti in classi binarie se necessario
    pred_classes <- ifelse(predictions > 1.5, 2, 1)
    
    # Calcola accuracy
    accuracies[i] <- mean(pred_classes == test_data[[target_col]])
  }
  
  return(mean(accuracies))
}

# Funzione completa per calcolare accuracy media dell'ensemble
calculate_ensemble_accuracy <- function(models, test_data, target_col) {
  
  # Vettore per memorizzare le accuracy individuali
  individual_accuracies <- numeric(length(models))
  
  # Lista per memorizzare tutte le predizioni
  all_predictions <- list()
  
  # Calcola predizioni per ogni modello
  for(i in 1:length(models)) {
    
    # Predizioni del modello i-esimo
    predictions <- predict(models[[i]], newdata = test_data)
    
    # Converti in classi binarie (assumendo che bnns restituisca probabilità o valori continui)
    # Se le predizioni sono già classi discrete, rimuovi questa conversione
    pred_classes <- ifelse(predictions > 1.5, 2, 1)  # soglia 1.5 per distinguere tra 1 e 2
    
    # Calcola accuracy per questo modello
    actual_values <- test_data[[target_col]]
    accuracy_i <- mean(pred_classes == actual_values)
    individual_accuracies[i] <- accuracy_i
    
    # Salva le predizioni
    all_predictions[[i]] <- pred_classes
    
    cat("Modello", i, "- Accuracy:", round(accuracy_i, 4), "\n")
  }
  
  # Calcola accuracy media dei modelli individuali
  mean_individual_accuracy <- mean(individual_accuracies)
  
  # Calcola anche ensemble prediction (media delle predizioni)
  ensemble_predictions <- Reduce("+", all_predictions) / length(all_predictions)
  ensemble_classes <- ifelse(ensemble_predictions > 1.5, 2, 1)
  ensemble_accuracy <- mean(ensemble_classes == test_data[[target_col]])
  
  # Restituisci risultati
  results <- list(
    individual_accuracies = individual_accuracies,
    mean_individual_accuracy = mean_individual_accuracy,
    ensemble_accuracy = ensemble_accuracy,
    ensemble_predictions = ensemble_classes
  )
  
  return(results)
}