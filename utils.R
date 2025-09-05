library(bnns)

####ENSEMBLE
# Funzione per creare ensemble che simula Monte Carlo sampling
create_uncertainty_ensemble <- function(data, target_col, n_models = 10) {
  models <- list()
  
  # Crea formula dinamicamente
  formula_str <- paste(target_col, "~ .") # Su quale feature si allena e quale target
  
  for(i in 1:n_models) {
    cat("Training model", i, "of", n_models, "\n")
    
    # Bootstrap sampling (per aleatoric)
    # il secondo argomento è per l'esattezza delle righe
    boot_indices <- sample(nrow(data), nrow(data), replace = TRUE) 
    boot_data <- data[boot_indices, ]
    
    # Aggiungi noise ai parametri (per epistemic) 
    set.seed(i)  # diversi seed per diversi modelli
    
    # Parametri specifici per dataset sintetici (Sezione 5.2 del paper)
    hidden_layers <- NULL         # Nessun hidden layer
    epochs <- 5                   # 5 epoche
    
    # Allena BNN con bnns
    model <- bnns(
      formula = as.formula(formula_str),
      data = boot_data,
      hidden_layers = hidden_layers,
      epochs = epochs,
      prior_pi = 0.5,         # π = 0.5 probabilità che prensa una delle due distr dei pesi
      prior_sigma1 = 0,       # σ₁ = 0
      prior_sigma2 = 6,       # σ₂ = 6
      lambda = 2000,          # λ = 2000 regolarizzazione
      init_mu = 0,            # μ = 0
      init_sigma = 1,         # σ = 1
      early_stopping = TRUE,
      verbose = TRUE,         # messaggi di log
      seed = i
    )
    
    models[[i]] <- model
  }
  return(models)
}
####

#### RATEI TP, TN, FP, FN
calculate_ensemble_metrics_by_group <- function(models, test_data, target_col, group_col = "G",
                                              threshold = 1.5, positive_class = 2) {
  
  # Estrai le variabili principali
  y_true <- test_data[[target_col]]
  groups <- test_data[[group_col]]
  unique_groups <- unique(groups)
  
  # Lista per salvare i risultati
  results <- list()
  
  # ===== 1. PREDIZIONI ENSEMBLE =====
  # Raccogli tutte le predizioni (gestendo BNN con campioni MC)
  all_predictions <- matrix(0, nrow = nrow(test_data), ncol = length(models))
  
  for (i in 1:length(models)) {
    preds_mc <- predict(models[[i]], test_data)
    # Media su tutti i campioni MC per ottenere le probabilità finali
    probabilities <- rowMeans(preds_mc)
    all_predictions[, i] <- ifelse(probabilities > threshold, 2, 1)
  }
  
  # Media delle predizioni (voto di maggioranza)
  ensemble_pred <- ifelse(rowMeans(all_predictions) > threshold, 2, 1)
  
  # ===== 2. METRICHE ENSEMBLE PER GRUPPO =====
  ensemble_results <- data.frame()
  
  for (group in unique_groups) {
    # Seleziona solo i dati di questo gruppo
    group_mask <- groups == group
    y_group <- y_true[group_mask]
    pred_group <- ensemble_pred[group_mask]
    
    # Calcola TP, TN, FP, FN per ensemble
    tp <- sum(y_group == positive_class & pred_group == positive_class)
    tn <- sum(y_group != positive_class & pred_group != positive_class)
    fp <- sum(y_group != positive_class & pred_group == positive_class)
    fn <- sum(y_group == positive_class & pred_group != positive_class)
    
    # Calcola metriche ensemble
    total <- tp + tn + fp + fn
    accuracy <- (tp + tn) / total
    tpr <- if (tp + fn > 0) tp / (tp + fn) else 0
    tnr <- if (tn + fp > 0) tn / (tn + fp) else 0
    fpr <- if (fp + tn > 0) fp / (fp + tn) else 0
    fnr <- if (fn + tp > 0) fn / (fn + tp) else 0
    ppv <- if (tp + fp > 0) tp / (tp + fp) else 0
    npv <- if (tn + fn > 0) tn / (tn + fn) else 0
    
    ensemble_row <- data.frame(
      group = group,
      accuracy = accuracy,
      tpr = tpr,
      tnr = tnr,
      fpr = fpr,
      fnr = fnr,
      ppv = ppv,
      npv = npv,
      tp = tp,
      tn = tn,
      fp = fp,
      fn = fn
    )
    
    ensemble_results <- rbind(ensemble_results, ensemble_row)
  }
  
  # ===== 3. CALCOLO FAIRNESS (assumendo 2 gruppi) =====
  fairness_results <- NULL
  
  if (length(unique_groups) == 2) {
    # Ordina i gruppi
    groups_sorted <- sort(unique_groups)
    minority_group <- groups_sorted[1]
    majority_group <- groups_sorted[2]
    
    # Estrai metriche per i due gruppi
    minority_metrics <- ensemble_results[ensemble_results$group == minority_group, ]
    majority_metrics <- ensemble_results[ensemble_results$group == majority_group, ]
    
    # Calcola total per denominatori
    minority_total <- minority_metrics$tp + minority_metrics$tn + minority_metrics$fp + minority_metrics$fn
    majority_total <- majority_metrics$tp + majority_metrics$tn + majority_metrics$fp + majority_metrics$fn
    
    # Calcola fairness measures
    fairness_results <- data.frame(
      minority = minority_group,
      majority = majority_group,
      F_SP = ((minority_metrics$tp + minority_metrics$fp) / minority_total) / 
        ((majority_metrics$tp + majority_metrics$fp) / majority_total),
      F_EOpp = minority_metrics$fnr / majority_metrics$fnr,
      F_EOdd = minority_metrics$tpr / majority_metrics$tpr,
      F_EAcc = minority_metrics$accuracy / majority_metrics$accuracy,
      TPR_ratio = minority_metrics$tpr / majority_metrics$tpr,
      TNR_ratio = minority_metrics$tnr / majority_metrics$tnr,
      FPR_ratio = minority_metrics$fpr / majority_metrics$fpr,
      FNR_ratio = minority_metrics$fnr / majority_metrics$fnr
    )
  }

  
  # ===== RESTITUISCI I RISULTATI =====
  return(list(
    ensemble_metrics_by_group = ensemble_results,
    fairness_measures = fairness_results
  ))
}

####


####INCERTEZZE
# Calcola uncertainties seguendo Eq. 7 del paper
calculate_uncertainties <- function(ensemble_models, test_data) {
  M <- length(ensemble_models)
  n_samples <- nrow(test_data)
  
  # Matrice delle predizioni: ogni riga = sample, ogni colonna = modello
  P_matrix <- matrix(0, nrow = n_samples, ncol = M)
  
  # Ottieni predizioni probabilistiche da ogni modello
  for(m in 1:M) {
    preds_mc <- predict(ensemble_models[[m]], newdata = test_data)
    preds <- rowMeans(preds_mc)  # Media su tutti i campioni dei modelli di MC
    P_matrix[, m] <- preds
  }
  
  # Calcola P̄ (media delle predizioni)
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
      aleatoric[i] <- aleatoric[i] + (p_m - p_m^2)  # Varianza binaria
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
    group_indices <- which(test_data[[sensitive_attr]] == group) # which() in R restituisce gli indici
    
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
####