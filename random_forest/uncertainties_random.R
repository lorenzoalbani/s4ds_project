# Calcola uncertainties seguendo Eq. 7 del paper
calculate_uncertainties <- function(ensemble_models, test_data) {
  M <- length(ensemble_models)
  n_samples <- nrow(test_data)
  
  # Matrice delle predizioni: ogni riga = sample, ogni colonna = modello
  P_matrix <- matrix(0, nrow = n_samples, ncol = M)
  
  # Ottieni predizioni probabilistiche da ogni modello
  for(m in 1:M) {
    # Per classificazione binaria, prendi prob classe positiva
    preds <- predict(ensemble_models[[m]], test_data, type = "prob")[,2]
    P_matrix[,m] <- preds
  }
  
  # Calcola P̄ (media delle predizioni)
  P_bar <- rowMeans(P_matrix)
  
  # EPISTEMIC UNCERTAINTY (Eq. 7 prima parte)
  epistemic <- numeric(n_samples)
  for(i in 1:n_samples) {
    diff <- P_matrix[i,] - P_bar[i]
    epistemic[i] <- mean(diff^2)  # (P_m - P̄)^T * (P_m - P̄)
  }
  
  # ALEATORIC UNCERTAINTY (Eq. 7 seconda parte)  
  aleatoric <- numeric(n_samples)
  for(i in 1:n_samples) {
    for(m in 1:M) {
      p_m <- P_matrix[i,m]
      # diag(P_m) - P_m^T * P_m per classificazione binaria
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

# Funzione per calcolare media uncertainties per gruppo
calculate_group_uncertainties <- function(uncertainties, sensitive_attr) {
  
  # Identifica i gruppi
  groups <- unique(test_data[[sensitive_attr]])
  
  # Risultato finale
  results <- data.frame(
    Group = character(),
    Mean_Epistemic = numeric(),
    Mean_Aleatoric = numeric(), 
    Mean_Predictive = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (group in groups) {
    # Indici dei campioni di questo gruppo
    group_indices <- which(test_data[[sensitive_attr]] == group)
    
    # Calcola medie delle uncertainties per questo gruppo
    mean_epistemic <- mean(uncertainties$epistemic[group_indices])
    mean_aleatoric <- mean(uncertainties$aleatoric[group_indices])
    mean_predictive <- mean(uncertainties$predictive[group_indices])
    
    # Aggiungi ai risultati
    results <- rbind(results, data.frame(
      Group = as.character(group),
      Mean_Epistemic = mean_epistemic,
      Mean_Aleatoric = mean_aleatoric,
      Mean_Predictive = mean_predictive
    ))
    
    # Stampa risultati
    cat("Gruppo", group, ":\n")
    cat("  Epistemic Uncertainty:", round(mean_epistemic, 6), "\n")
    cat("  Aleatoric Uncertainty:", round(mean_aleatoric, 6), "\n") 
    cat("  Predictive Uncertainty:", round(mean_predictive, 6), "\n\n")
  }
  
  return(results)
}

