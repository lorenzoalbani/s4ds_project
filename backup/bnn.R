library(bnns)
library(dplyr)

# Funzione per creare BNN per dataset sintetici secondo il paper
create_uncertainty_bnn <- function(data, target_col) {
  
  # Parametri specifici per dataset sintetici (Sezione 5.2 del paper)
  hidden_layers <- NULL         # Nessun hidden layer
  epochs <- 5                   # 5 epoche
  batch_size <- 8               # batch size 8
  n_monte_carlo <- 10           # T = 10 campioni Monte Carlo
  
  # Formula dinamica
  formula_str <- paste(target_col, "~ .")
  
  cat("Training BNN for synthetic dataset with", epochs, "epochs\n")
  
  # Modello BNN con Bayes by Backprop
  model <- bnns(
    formula = as.formula(formula_str),
    data = data,
    hidden_layers = hidden_layers,
    epochs = epochs,
    prior_pi = 0.5,         # π = 0.5
    prior_sigma1 = 0,       # σ₁ = 0
    prior_sigma2 = 6,       # σ₂ = 6
    lambda = 2000,          # λ = 2000
    init_mu = 0,            # μ = 0
    init_sigma = 1,         # σ = 1
    early_stopping = TRUE,
    verbose = TRUE
  )
  
  return(model)
}

# Funzione per quantificare incertezza secondo Eq. (7) del paper
quantify_uncertainty <- function(model, newdata, n_samples = 10) {
  
  # Genera T=10 predizioni Monte Carlo
  mc_predictions <- matrix(0, nrow = nrow(newdata), ncol = n_samples)
  
  for(t in 1:n_samples) {
    # Sampling da posterior dei pesi
    pred_t <- predict(model, newdata = newdata, type = "response", 
                      mc_sample = TRUE, seed = t)
    mc_predictions[, t] <- pred_t
  }
  
  # Calcola P_bar (media delle predizioni)
  P_bar <- rowMeans(mc_predictions)
  
  # Calcola incertezza epistemica (Eq. 7)
  epistemic_unc <- numeric(nrow(newdata))
  for(i in 1:nrow(newdata)) {
    diff_squared <- (mc_predictions[i, ] - P_bar[i])^2
    epistemic_unc[i] <- mean(diff_squared)
  }
  
  # Calcola incertezza aleatoric (Eq. 7) 
  aleatoric_unc <- numeric(nrow(newdata))
  for(i in 1:nrow(newdata)) {
    for(t in 1:n_samples) {
      P_t <- mc_predictions[i, t]
      aleatoric_unc[i] <- aleatoric_unc[i] + (P_t * (1 - P_t))
    }
    aleatoric_unc[i] <- aleatoric_unc[i] / n_samples
  }
  
  # Incertezza predittiva totale
  predictive_unc <- epistemic_unc + aleatoric_unc
  
  return(list(
    epistemic = epistemic_unc,
    aleatoric = aleatoric_unc,
    predictive = predictive_unc,
    predictions = P_bar,
    mc_predictions = mc_predictions
  ))
}

# Funzione per calcolare fairness measures secondo Eq. (27)
calculate_fairness_measures <- function(uncertainty_results, groups, minority_group = 0) {
  
  # Separa gruppi
  minority_mask <- groups == minority_group
  majority_mask <- groups != minority_group
  
  # Media incertezze per gruppo
  U_minority <- list(
    epistemic = mean(uncertainty_results$epistemic[minority_mask]),
    aleatoric = mean(uncertainty_results$aleatoric[minority_mask]),
    predictive = mean(uncertainty_results$predictive[minority_mask])
  )
  
  U_majority <- list(
    epistemic = mean(uncertainty_results$epistemic[majority_mask]),
    aleatoric = mean(uncertainty_results$aleatoric[majority_mask]),
    predictive = mean(uncertainty_results$predictive[majority_mask])
  )
  
  # Fairness ratios (Eq. 27): F_u = U(G=0)/U(G=1)
  fairness_measures <- list(
    F_Epis = U_minority$epistemic / U_majority$epistemic,
    F_Alea = U_minority$aleatoric / U_majority$aleatoric,
    F_Pred = U_minority$predictive / U_majority$predictive
  )
  
  # Unfair se |F - 1| > 0.2 (seguendo Feldman et al., 2015)
  fairness_status <- list(
    epistemic_fair = abs(fairness_measures$F_Epis - 1) <= 0.2,
    aleatoric_fair = abs(fairness_measures$F_Alea - 1) <= 0.2,
    predictive_fair = abs(fairness_measures$F_Pred - 1) <= 0.2
  )
  
  return(list(
    measures = fairness_measures,
    status = fairness_status,
    uncertainties = list(minority = U_minority, majority = U_majority)
  ))
}

# Esempio d'uso per dataset sintetico (come nel paper)
# create_synthetic_dataset_1 <- function() {
#   # Implementa le distribuzioni dalle Eq. (10-13) per SD1
#   # ... codice per generare dati sintetici
# }

# Usage example:
# data <- your_synthetic_dataset
# model <- create_uncertainty_bnn(data, "Y", "synthetic")
# uncertainty <- quantify_uncertainty(model, data)
# fairness <- calculate_fairness_measures(uncertainty, data$G)


#accuracy per gruppo
# Calcola accuracy per gruppo (per ogni modello e per l'ensemble)
# - models: lista di modelli BNN (come restituita da create_uncertainty_ensemble)
# - test_data: data.frame con almeno le colonne target_col e group_col
# - target_col: nome della colonna del target (es. 'y')
# - group_col: nome della colonna del gruppo sensibile (es. 'G')
# - threshold: soglia per convertire predizioni continue in classi {1,2}
# - levels_order: opzionale, vettore di 2 livelli che definisce (minority, majority)
#                 per calcolare F_EAcc = Acc(minority)/Acc(majority)
calculate_ensemble_accuracy_by_group <- function(models,
                                                 test_data,
                                                 target_col,
                                                 group_col = "G",
                                                 threshold = 1.5,
                                                 levels_order = NULL) {
  stopifnot(length(models) > 0)
  td <- test_data
  
  # Estraggo y e G
  y_true <- td[[target_col]]
  groups <- td[[group_col]]
  if (is.factor(groups)) groups <- droplevels(groups)
  
  # Predizioni per ogni modello (classi 1/2 via soglia)
  all_pred_classes <- vector("list", length(models))
  individual_acc_by_group <- list()
  
  for (i in seq_along(models)) {
    raw_pred <- predict(models[[i]], newdata = td)
    
    # Se predict() restituisce un vettore continuo, uso la tua soglia 1.5
    # Se restituisce già classi, adegua qui (es. as.numeric/raw_pred).
    pred_cls <- ifelse(raw_pred > threshold, 2, 1)
    
    # Accuracy per gruppo (tibble base)
    acc_tbl <- aggregate((pred_cls == y_true) ~ groups, FUN = mean)
    names(acc_tbl) <- c("group", "accuracy")
    acc_tbl$model <- i
    
    individual_acc_by_group[[i]] <- acc_tbl
    all_pred_classes[[i]] <- pred_cls
  }
  
  # Tabella con accuracy per gruppo e per modello
  individual_acc_by_group_df <- do.call(rbind, individual_acc_by_group)
  
  # Media delle accuracy dei modelli per ciascun gruppo
  mean_individual_acc_by_group <- aggregate(accuracy ~ group, data = individual_acc_by_group_df, FUN = mean)
  
  # Predizione di ensemble (majority/mean voting su {1,2})
  # Con media > 1.5 => classe 2, altrimenti 1
  pred_mat <- do.call(cbind, all_pred_classes)
  ensemble_cls <- ifelse(rowMeans(pred_mat) > threshold, 2, 1)
  
  # Accuracy dell'ensemble per gruppo
  ens_acc_by_group <- aggregate((ensemble_cls == y_true) ~ groups, FUN = mean)
  names(ens_acc_by_group) <- c("group", "ensemble_accuracy")
  
  # Calcolo Equal Accuracy: F_EAcc = Acc(G=minority)/Acc(G=majority) (Eq. 26)
  # Se non fornisci levels_order, uso l'ordinamento naturale dei livelli presenti
  # (consigliato però specificare esplicitamente minority/majority).
  if (is.null(levels_order)) {
    uniq_groups <- as.character(sort(unique(groups)))
    if (length(uniq_groups) != 2) {
      stop("Per F_EAcc servono esattamente due gruppi; specifica `levels_order` o riduci a due livelli.")
    }
    levels_order <- uniq_groups
  }
  # Prendo le accuracy ensemble nei due gruppi nell’ordine indicato
  acc_lookup <- setNames(ens_acc_by_group$ensemble_accuracy, as.character(ens_acc_by_group$group))
  if (!all(levels_order %in% names(acc_lookup))) {
    stop("I livelli di `levels_order` non combaciano con i valori osservati in `group_col`.")
  }
  F_EAcc <- acc_lookup[[ levels_order[1] ]] / acc_lookup[[ levels_order[2] ]]
  
  list(
    individual_accuracies_by_group = individual_acc_by_group_df,  # per modello e gruppo
    mean_individual_accuracy_by_group = mean_individual_acc_by_group,  # media sui modelli
    ensemble_accuracy_by_group = ens_acc_by_group,  # ensemble per gruppo
    F_EAcc = data.frame(minority = levels_order[1],
                        majority = levels_order[2],
                        F_EAcc = F_EAcc)  # Eq. 26 dell’articolo
  )
}


# metrics_functions.R
# Functions for calculating classification metrics by group for BNN ensemble

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
    pred <- ifelse(predict(models[[i]], newdata = test_data) > threshold, 2, 1)
    do.call(rbind, lapply(unique_groups, function(g) {
      m <- groups == g
      cbind(model = i, group = g, get_metrics(y_true[m], pred[m]))
    }))
  })
  
  individual_df <- do.call(rbind, individual_metrics)
  mean_metrics <- aggregate(. ~ group, data = individual_df[, -1], FUN = mean)
  
  pred_mat <- do.call(cbind, lapply(models, function(m) ifelse(predict(m, newdata = test_data) > threshold, 2, 1)))
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


calculate_metrics_by_group_2 <- function(data, group_col = "G", true_col = "true", pred_col = "pred", threshold = 0.5) {
  library(dplyr)
  
  # Binarizza le predizioni
  data <- data %>%
    mutate(pred_bin = ifelse(.data[[pred_col]] >= threshold, 1, 0),
           true_bin = ifelse(.data[[true_col]] >= threshold, 1, 0))
  
  # Funzione per calcolare le metriche
  compute_metrics <- function(df) {
    TP <- sum(df$pred_bin == 1 & df$true_bin == 1)
    TN <- sum(df$pred_bin == 0 & df$true_bin == 0)
    FP <- sum(df$pred_bin == 1 & df$true_bin == 0)
    FN <- sum(df$pred_bin == 0 & df$true_bin == 1)
    total <- nrow(df)
    
    accuracy <- (TP + TN) / total
    TPR <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    TNR <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
    FPR <- ifelse((FP + TN) > 0, FP / (FP + TN), NA)
    FNR <- ifelse((FN + TP) > 0, FN / (FN + TP), NA)
    
    return(data.frame(accuracy, TPR, TNR, FPR, FNR))
  }
  
  # Applica per gruppo
  metrics_by_group <- data %>%
    group_by(.data[[group_col]]) %>%
    group_modify(~ compute_metrics(.x)) %>%
    ungroup()
  
  return(metrics_by_group)
}

