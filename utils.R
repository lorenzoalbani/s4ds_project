library(bnns)

####ENSEMBLE
# Funzione per creare ensemble di BNN che simula Monte Carlo sampling
create_uncertainty_ensemble <- function(data, target_col, n_models = 10) { #dati train, target, #modelli
  models <- list()
  
  # Crea formula dinamicamente
  formula_str <- paste(target_col, "~ .") # Quale target(y) e su quale feature si allena(tutte le altre)
   #restituisce stringa
  for(i in 1:n_models) { # loop per allenare modelli 
    cat("Training model", i, "of", n_models, "\n") # print
    
    # Bootstrap sampling (per aleatoric)
    # il secondo argomento è per l'esattezza delle righe
    boot_indices <- sample(nrow(data), nrow(data), replace = TRUE) #indici da data con ripescaggio
    boot_data <- data[boot_indices, ] #creo dataset
    #ogni modello sarà allenato su dati diversi creati da bootstrap diversi
    
    # Aggiungi noise ai parametri (per epistemic) 
    set.seed(i)  # diversi seed per diversi modelli
    
    # Parametri specifici per dataset sintetici (Sezione 5.2 del paper)
    hidden_layers <- NULL         # Nessun hidden layer
    epochs <- 5                   # 5 epoche
    
    # Allena BNN con bnns
    model <- bnns(
      formula = as.formula(formula_str), #variabili su cui si allena e quale è la target 
      data = boot_data,
      hidden_layers = hidden_layers,
      epochs = epochs,
      prior_pi = 0.5,         # π = 0.5 probabilità che prenda una delle due distr dei pesi
      prior_sigma1 = 0,       # σ₁ = 0
      prior_sigma2 = 6,       # σ₂ = 6
      lambda = 2000,          # λ = 2000 regolarizzazione
      init_mu = 0,            # μ = 0
      init_sigma = 1,         # σ = 1
      early_stopping = TRUE,
      verbose = TRUE,         # messaggi di log
      seed = i
    )
    
    models[[i]] <- model #inserisco nella lista modelli 
  }
  return(models) #return dell'ensemble
}
####

#### RATEI TP, TN, FP, FN per gruppo e misure di fairness
calculate_ensemble_metrics_by_group <- function(models, test_data, target_col, group_col = "G",
                                              threshold = 1.5, positive_class = 2) {
  # ensemble, dati test, target, gruppo, threshold, etichetta positiva
  # Estrai le variabili principali
  y_true <- test_data[[target_col]] # etichette vere test set
  groups <- test_data[[group_col]] # gruppi per riga
  unique_groups <- unique(groups) 
  
  # Lista per salvare i risultati
  results <- list()
    
  # ===== 1. PREDIZIONI ENSEMBLE =====
  # Raccogli tutte le predizioni (gestendo BNN con campioni MC)
  all_predictions <- matrix(0, nrow = nrow(test_data), ncol = length(models))
  
  for (i in 1:length(models)) { # loop sui modelli
    preds_mc <- predict(models[[i]], test_data) # campioni MC di output per ogni osservazione
    # Media per ottenere le probabilità finali della classe positiva
    probabilities <- rowMeans(preds_mc)
    #assegno classi
    all_predictions[, i] <- ifelse(probabilities > threshold, 2, 1)
  }
  
  # Media delle predizioni (voto di maggioranza)
  ensemble_pred <- ifelse(rowMeans(all_predictions) > threshold, 2, 1)
  
  # ===== 2. METRICHE ENSEMBLE PER GRUPPO =====
  ensemble_results <- data.frame()
  
  for (group in unique_groups) { # loop sui gruppi
    # Seleziona solo i dati di questo gruppo
    group_mask <- groups == group # maschera per le righe appartenenti al gruppo
    y_group <- y_true[group_mask] # etichette vere del gruppo
    pred_group <- ensemble_pred[group_mask] #predizione per quel gruppo
    
    # Calcola TP, TN, FP, FN per ensemble
    tp <- sum(y_group == positive_class & pred_group == positive_class)
    tn <- sum(y_group != positive_class & pred_group != positive_class)
    fp <- sum(y_group != positive_class & pred_group == positive_class)
    fn <- sum(y_group == positive_class & pred_group != positive_class)
    
    # Calcola metriche ensemble
    total <- tp + tn + fp + fn
    accuracy <- (tp + tn) / total
    # Evitiamo la divisione per 0
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
    # Accoda risultati
    ensemble_results <- rbind(ensemble_results, ensemble_row)
  }
  
  # ===== 3. CALCOLO FAIRNESS (assumendo 2 gruppi) =====
  fairness_results <- NULL
  
  if (length(unique_groups) == 2) {
    # Ordina i gruppi
    groups_sorted <- sort(unique_groups)
    minority_group <- groups_sorted[1] # G0 minoritario
    majority_group <- groups_sorted[2] # G1 maggioritario
    
    # Estrai metriche per i due gruppi
    minority_metrics <- ensemble_results[ensemble_results$group == minority_group, ]
    majority_metrics <- ensemble_results[ensemble_results$group == majority_group, ]
    
    # Calcola osservazioni totali per gruppo
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
    ensemble_metrics_by_group = ensemble_results, #tabella metriche per gruppo
    fairness_measures = fairness_results # tabella misure di fairness
  ))
}

####


####INCERTEZZE
# Calcola uncertainties seguendo Eq. 7 del paper
calculate_uncertainties <- function(ensemble_models, test_data) {
  # lista modelli BNN, dati test
  M <- length(ensemble_models) #modelli
  n_samples <- nrow(test_data) #righe test set
  
  # Matrice delle predizioni classe positiva: ogni riga = sample, ogni colonna = modello
  P_matrix <- matrix(0, nrow = n_samples, ncol = M)
  
  # Ottieni predizioni probabilistiche da ogni modello
  for(m in 1:M) { # loop modelli
    preds_mc <- predict(ensemble_models[[m]], newdata = test_data) #m campioni MC per ogni riga
    preds <- rowMeans(preds_mc)  # Media su tutti i campioni dei modelli di MC
    P_matrix[, m] <- preds #Assegna P alla colonna m
  }
  
  # Calcola P̄ (media delle predizioni)
  P_bar <- rowMeans(P_matrix)
  
  # EPISTEMIC UNCERTAINTY (Eq. 7 prima parte)
  epistemic <- numeric(n_samples)
  #varianza tra modelli delle probabilità per ciascun sample
  for(i in 1:n_samples) {
    diff <- P_matrix[i,] - P_bar[i]
    epistemic[i] <- mean(diff^2)
  }
  
  # ALEATORIC UNCERTAINTY (Eq. 7 seconda parte)  
  aleatoric <- numeric(n_samples)
  for(i in 1:n_samples) {
    sum_aleatoric <- 0
    for(m in 1:M) {
      p_m <- P_matrix[i,m]
      # Formula corretta per caso binario: diag(P_m) - P_m^T * P_m
      # = 1 - (p² + (1-p)²) = 2p(1-p)
      sum_aleatoric <- sum_aleatoric + 2 * p_m * (1 - p_m)
    }
    aleatoric[i] <- sum_aleatoric / M
  }
  
  # PREDICTIVE UNCERTAINTY = Epistemic + Aleatoric
  predictive <- epistemic + aleatoric
  
  return(list( #vettori lunghezza n_samples
    epistemic = epistemic,
    aleatoric = aleatoric, 
    predictive = predictive
  ))
}


# Funzione per calcolare media uncertainties per gruppo
calculate_group_uncertainties <- function(uncertainties, sensitive_attr) {
  # lista uncertanties, gruppo
  # Identifica i gruppi
  groups <- unique(test_data[[sensitive_attr]])
  
  # Risultato finale
  results <- data.frame(
    Group = character(),
    Mean_Epistemic = numeric(),
    Mean_Aleatoric = numeric(), 
    Mean_Predictive = numeric(),
    stringsAsFactors = FALSE #evita conversione automatica 
  )
  
  for (group in groups) { #loop sui gruppi
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