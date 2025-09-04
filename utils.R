library(bnns)

####ENSEMBLE
# Funzione per creare ensemble che simula Monte Carlo sampling
create_uncertainty_ensemble <- function(data, target_col, n_models = 10) {
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
    
    # Allena BNN con bnns
    model <- bnns(
      formula = as.formula(formula_str),
      data = boot_data,
      hidden_layers = 0,
      epochs = 5,
      seed = i
    )
    
    models[[i]] <- model
  }
  return(models)
}
####

#### ACCURACY
# Funzione semplice per calcolare accuracy media (viene usata in funz dopo)
calculate_mean_accuracy <- function(models, test_data, target_col) {
  
  accuracies <- numeric(length(models))
  
  for(i in 1:length(models)) {
    # Predizioni probabilistiche MC della BNN
    preds_mc <- predict(models[[i]], newdata = test_data)
    
    # Media su tutti i campioni MC
    probabilities <- rowMeans(preds_mc)
    
    # Converti in classi binarie
    pred_classes <- ifelse(probabilities > 0.5, 1, 0)
    
    # Calcola accuracy
    accuracies[i] <- mean(pred_classes == test_data[[target_col]])
  }
  
  return(mean(accuracies))
}

calculate_group_accuracy_simple <- function(models, test_data, target_col, group_col) {
  groups <- unique(test_data[[group_col]])
  results <- list()
  
  for(group in groups) {
    group_data <- test_data[test_data[[group_col]] == group, ]
    acc <- calculate_mean_accuracy(models, group_data, target_col)
    results[[group]] <- acc
  }
  
  return(results)
}
####


#### RATEI TP, TN, FP, FN
calculate_ensemble_metrics_by_group <- function(models, test_data, target_col, group_col = "G",
                                                threshold = 1.5, levels_order = NULL, positive_class = 2) {
  stopifnot(length(models) > 0)
  y_true <- test_data[[target_col]]
  groups <- droplevels(as.factor(test_data[[group_col]]))
  unique_groups <- unique(groups)
  
  get_metrics <- function(y, pred) {
    tp <- sum(y == positive_class & pred == positive_class)
    tn <- sum(y != positive_class & pred != positive_class)
    fp <- sum(y != positive_class & pred == positive_class)
    fn <- sum(y == positive_class & pred != positive_class)
    total <- tp + tn + fp + fn
    data.frame(
      accuracy = (tp + tn) / total,
      tpr = if (tp + fn > 0) tp / (tp + fn) else 0,
      tnr = if (tn + fp > 0) tn / (tn + fp) else 0,
      fpr = if (fp + tn > 0) fp / (fp + tn) else 0,
      fnr = if (fn + tp > 0) fn / (fn + tp) else 0,
      ppv = if (tp + fp > 0) tp / (tp + fp) else 0,
      npv = if (tn + fn > 0) tn / (tn + fn) else 0,
      tp = tp, tn = tn, fp = fp, fn = fn
    )
  }
  
  individual_metrics <- lapply(seq_along(models), function(i) {
    pred <- ifelse(predict(models[[i]], test_data) > threshold, 2, 1)
    do.call(rbind, lapply(unique_groups, function(g) {
      m <- groups == g
      cbind(model = i, group = g, get_metrics(y_true[m], pred[m]))
    }))
  })
  
  individual_df <- do.call(rbind, individual_metrics)
  mean_metrics <- aggregate(. ~ group, data = individual_df[, -1], FUN = mean)
  
  pred_mat <- do.call(cbind, lapply(models, function(m) ifelse(predict(m, test_data) > threshold, 2, 1)))
  ensemble_pred <- ifelse(rowMeans(pred_mat) > threshold, 2, 1)
  
  ensemble_metrics <- do.call(rbind, lapply(unique_groups, function(g) {
    m <- groups == g
    cbind(group = g, get_metrics(y_true[m], ensemble_pred[m]))
  }))
  
  if (is.null(levels_order)) {
    levels_order <- sort(as.character(unique(groups)))
    stopifnot(length(levels_order) == 2)
  }
  
  ens_lookup <- split(ensemble_metrics, ensemble_metrics$group)
  minority <- ens_lookup[[levels_order[1]]]
  majority <- ens_lookup[[levels_order[2]]]
  
  fairness_measures <- data.frame(
    minority = levels_order[1], majority = levels_order[2],
    F_SP = ((minority$tp + minority$fp) / sum(minority[c("tp", "tn", "fp", "fn")])) /
      ((majority$tp + majority$fp) / sum(majority[c("tp", "tn", "fp", "fn")])),
    F_EOpp = minority$fnr / majority$fnr,
    F_EOdd = minority$tpr / majority$tpr,
    F_EAcc = minority$accuracy / majority$accuracy,
    TPR_ratio = minority$tpr / majority$tpr,
    TNR_ratio = minority$tnr / majority$tnr,
    FPR_ratio = minority$fpr / majority$fpr,
    FNR_ratio = minority$fnr / majority$fnr
  )
  
  list(
    individual_metrics_by_group = individual_df,
    mean_individual_metrics_by_group = mean_metrics,
    ensemble_metrics_by_group = ensemble_metrics,
    fairness_measures = fairness_measures,
    confusion_matrices = list(
      minority = minority[c("tp", "tn", "fp", "fn")],
      majority = majority[c("tp", "tn", "fp", "fn")]
    )
  )
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
    # Media su tutti i campioni MC
    preds <- rowMeans(preds_mc)
    P_matrix[, m] <- preds
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
####