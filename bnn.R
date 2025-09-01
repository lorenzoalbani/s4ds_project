library(bnns)  # Bayesian Neural Networks
library(dplyr)

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
    
    # SOSTITUITO: Allena BNN con bnns
    model <- bnns(
      formula = as.formula(formula_str),
      data = boot_data,
      hidden_layers = 0,  # varia architettura
      epochs = 5,
      seed = i
    )
    
    models[[i]] <- model
  }
  return(models)
}


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

calculate_ensemble_metrics_by_group <- function(models,
                                                test_data,
                                                target_col,
                                                group_col = "G",
                                                threshold = 1.5,
                                                levels_order = NULL,
                                                positive_class = 2) {
  stopifnot(length(models) > 0)
  td <- test_data
  
  # Extract y and G
  y_true <- td[[target_col]]
  groups <- td[[group_col]]
  if (is.factor(groups)) groups <- droplevels(groups)
  
  # Individual model predictions
  all_pred_classes <- vector("list", length(models))
  individual_metrics_by_group <- list()
  
  for (i in seq_along(models)) {
    raw_pred <- predict(models[[i]], newdata = td)
    pred_cls <- ifelse(raw_pred > threshold, 2, 1)
    
    # Calculate metrics for each group
    unique_groups <- unique(groups)
    group_metrics <- data.frame()
    
    for (g in unique_groups) {
      mask <- groups == g
      y_g <- y_true[mask]
      pred_g <- pred_cls[mask]
      
      # Confusion matrix components
      tp <- sum(y_g == positive_class & pred_g == positive_class)
      tn <- sum(y_g != positive_class & pred_g != positive_class)
      fp <- sum(y_g != positive_class & pred_g == positive_class)
      fn <- sum(y_g == positive_class & pred_g != positive_class)
      
      # Calculate rates
      tpr <- if (tp + fn > 0) tp / (tp + fn) else 0  # True Positive Rate
      tnr <- if (tn + fp > 0) tn / (tn + fp) else 0  # True Negative Rate
      fpr <- if (fp + tn > 0) fp / (fp + tn) else 0  # False Positive Rate
      fnr <- if (fn + tp > 0) fn / (fn + tp) else 0  # False Negative Rate
      
      # Additional metrics from paper
      accuracy <- (tp + tn) / (tp + tn + fp + fn)
      ppv <- if (tp + fp > 0) tp / (tp + fp) else 0  # Positive Predictive Value
      npv <- if (tn + fn > 0) tn / (tn + fn) else 0  # Negative Predictive Value
      
      group_metrics <- rbind(group_metrics, data.frame(
        model = i,
        group = g,
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
      ))
    }
    
    individual_metrics_by_group[[i]] <- group_metrics
    all_pred_classes[[i]] <- pred_cls
  }
  
  # Combine individual model results
  individual_metrics_df <- do.call(rbind, individual_metrics_by_group)
  
  # Mean metrics across models for each group
  mean_metrics <- aggregate(
    cbind(accuracy, tpr, tnr, fpr, fnr, ppv, npv) ~ group, 
    data = individual_metrics_df, 
    FUN = mean
  )
  
  # Ensemble predictions (majority voting)
  pred_mat <- do.call(cbind, all_pred_classes)
  ensemble_cls <- ifelse(rowMeans(pred_mat) > threshold, 2, 1)
  
  # Ensemble metrics by group
  unique_groups <- unique(groups)
  ensemble_metrics <- data.frame()
  
  for (g in unique_groups) {
    mask <- groups == g
    y_g <- y_true[mask]
    pred_g <- ensemble_cls[mask]
    
    # Confusion matrix
    tp <- sum(y_g == positive_class & pred_g == positive_class)
    tn <- sum(y_g != positive_class & pred_g != positive_class)
    fp <- sum(y_g != positive_class & pred_g == positive_class)
    fn <- sum(y_g == positive_class & pred_g != positive_class)
    
    # Rates
    tpr <- if (tp + fn > 0) tp / (tp + fn) else 0
    tnr <- if (tn + fp > 0) tn / (tn + fp) else 0
    fpr <- if (fp + tn > 0) fp / (fp + tn) else 0
    fnr <- if (fn + tp > 0) fn / (fn + tp) else 0
    
    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    ppv <- if (tp + fp > 0) tp / (tp + fp) else 0
    npv <- if (tn + fn > 0) tn / (tn + fn) else 0
    
    ensemble_metrics <- rbind(ensemble_metrics, data.frame(
      group = g,
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
    ))
  }
  
  # Calculate fairness measures (following paper's Section 5.3)
  if (is.null(levels_order)) {
    uniq_groups <- as.character(sort(unique(groups)))
    if (length(uniq_groups) != 2) {
      stop("Need exactly two groups for fairness measures")
    }
    levels_order <- uniq_groups
  }
  
  # Create lookup for ensemble metrics
  ens_lookup <- setNames(
    split(ensemble_metrics, ensemble_metrics$group),
    ensemble_metrics$group
  )
  
  if (!all(levels_order %in% names(ens_lookup))) {
    stop("levels_order values not found in group data")
  }
  
  minority_metrics <- ens_lookup[[levels_order[1]]]
  majority_metrics <- ens_lookup[[levels_order[2]]]
  
  # Fairness measures following paper's equations (23-26)
  
  # Calculate prediction rates
  pred_pos_minority <- (minority_metrics$tp + minority_metrics$fp) / 
    sum(minority_metrics[c("tp", "tn", "fp", "fn")])
  pred_pos_majority <- (majority_metrics$tp + majority_metrics$fp) / 
    sum(majority_metrics[c("tp", "tn", "fp", "fn")])
  
  # Statistical Parity (Eq. 23): P(Ŷ=1|G=0) / P(Ŷ=1|G=1)
  F_SP <- pred_pos_minority / pred_pos_majority
  
  # Equal Opportunity (Eq. 24): P(Ŷ=0|Y=1,G=0) / P(Ŷ=0|Y=1,G=1)
  F_EOpp <- minority_metrics$fnr / majority_metrics$fnr
  
  # Equalized Odds (Eq. 25): P(Ŷ=1|Y=y,G=0) / P(Ŷ=1|Y=y,G=1)
  # This should be calculated for both Y=0 and Y=1, here we use TPR (Y=1 case)
  F_EOdd <- minority_metrics$tpr / majority_metrics$tpr
  
  # Equal Accuracy (Eq. 26): Acc(G=0) / Acc(G=1)
  F_EAcc <- minority_metrics$accuracy / majority_metrics$accuracy
  
  fairness_measures <- data.frame(
    minority = levels_order[1],
    majority = levels_order[2],
    F_SP = F_SP,
    F_EOpp = F_EOpp,
    F_EOdd = F_EOdd,
    F_EAcc = F_EAcc,
    # Additional ratios for completeness
    TPR_ratio = minority_metrics$tpr / majority_metrics$tpr,
    TNR_ratio = minority_metrics$tnr / majority_metrics$tnr,
    FPR_ratio = minority_metrics$fpr / majority_metrics$fpr,
    FNR_ratio = minority_metrics$fnr / majority_metrics$fnr
  )
  
  list(
    individual_metrics_by_group = individual_metrics_df,
    mean_individual_metrics_by_group = mean_metrics,
    ensemble_metrics_by_group = ensemble_metrics,
    fairness_measures = fairness_measures,
    confusion_matrices = list(
      minority = minority_metrics[c("tp", "tn", "fp", "fn")],
      majority = majority_metrics[c("tp", "tn", "fp", "fn")]
    )
  )
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
