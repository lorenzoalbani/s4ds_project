library(torch)

# Funzione per creare ensemble BNN con PyTorch
create_uncertainty_ensemble <- function(data, target_col, n_models = 2) {
  models <- list()
  
  # Prepara i dati
  feature_cols <- setdiff(names(data), target_col)
  X <- as.matrix(data[, feature_cols])
  y <- as.numeric(data[[target_col]])  # Mantieni 1,2 per torch R
  
  # Converti in tensori
  X_tensor <- torch_tensor(X, dtype = torch_float())
  y_tensor <- torch_tensor(y, dtype = torch_long())
  
  # Parametri della rete
  input_size <- ncol(X)
  hidden_sizes <- c(20, 10)  # Puoi modificare l'architettura
  output_size <- 2  # Classificazione binaria
  
  for(i in 1:n_models) {
    set.seed(i)
    
    # Crea modello BNN
    model <- BayesianNN(
      input_size = input_size,
      hidden_sizes = hidden_sizes,
      output_size = output_size,
      prior_params = list(sigma1 = 0, sigma2 = 6, pi = 0.5)
    )
    
    # Crea dataset e dataloader
    dataset <- dataset(
      name = "tabular_dataset",
      initialize = function(X, y) {
        self$X <- X
        self$y <- y
      },
      .getitem = function(i) {
        list(x = self$X[i, ], y = self$y[i])
      },
      .length = function() {
        self$X$size(1)
      }
    )(X_tensor, y_tensor)
    
    # Bootstrap sampling per variabilità
    n_samples <- nrow(data)
    boot_indices <- sample(1:n_samples, n_samples, replace = TRUE)
    
    boot_X <- X_tensor[boot_indices, ]
    boot_y <- y_tensor[boot_indices]
    
    boot_dataset <- dataset(
      name = "tabular_dataset",
      initialize = function(X, y) {
        self$X <- X
        self$y <- y
      },
      .getitem = function(i) {
        list(x = self$X[i, ], y = self$y[i])
      },
      .length = function() {
        self$X$size(1)
      }
    )(boot_X, boot_y)
    
    # Dataloader
    train_loader <- dataloader(boot_dataset, batch_size = min(32, n_samples), shuffle = TRUE)
    
    # Training con parametri variabili per diversificare i modelli
    epochs <- sample(50:150, 1)
    lr <- runif(1, 0.0001, 0.01)
    lambda <- sample(c(1000, 2000, 5000), 1)
    
    # Addestra il modello
    model <- train_bayesian_nn(
      model = model,
      train_loader = train_loader,
      epochs = epochs,
      lr = lr,
      lambda = lambda
    )
    
    models[[i]] <- model
    cat("Modello", i, "addestrato con epochs =", epochs, ", lr =", round(lr, 4), "\n")
  }
  
  return(models)
}


# Funzione per fare predizioni con uncertainty
predict_ensemble <- function(models, newdata, n_samples = 10) {
  feature_names <- names(newdata)[!names(newdata) %in% c("y", "Y")]
  X_new <- as.matrix(newdata[, feature_names])
  X_tensor <- torch_tensor(X_new, dtype = torch_float())
  
  all_predictions <- list()
  
  for(i in 1:length(models)) {
    model <- models[[i]]
    model$eval()
    
    # Predizioni con incertezza
    with_no_grad({
      pred_results <- model$predict_with_uncertainty(X_tensor, n_samples = n_samples)
      # Prendi la classe più probabile dalla media
      class_probs <- as.array(pred_results$mean$cpu())
      predicted_classes <- apply(class_probs, 1, which.max) - 1  # Torna a 0,1
      predicted_classes <- predicted_classes + 1  # Converti a 1,2
    })
    
    all_predictions[[i]] <- predicted_classes
  }
  
  return(all_predictions)
}

# Funzione aggiornata per calcolare accuracy
calculate_ensemble_accuracy <- function(models, test_data, target_col) {
  
  # Ottieni predizioni
  all_predictions <- predict_ensemble(models, test_data)
  
  # Calcola accuracy individuali
  individual_accuracies <- numeric(length(models))
  actual_values <- test_data[[target_col]]
  
  for(i in 1:length(models)) {
    individual_accuracies[i] <- mean(all_predictions[[i]] == actual_values)
    cat("Modello", i, "- Accuracy:", round(individual_accuracies[i], 4), "\n")
  }
  
  # Calcola ensemble prediction (moda)
  ensemble_predictions <- apply(do.call(cbind, all_predictions), 1, function(x) {
    tbl <- table(x)
    as.numeric(names(tbl)[which.max(tbl)])
  })
  
  # Accuracy dell'ensemble
  ensemble_accuracy <- mean(ensemble_predictions == actual_values)
  
  results <- list(
    individual_accuracies = individual_accuracies,
    mean_individual_accuracy = mean(individual_accuracies),
    ensemble_accuracy = ensemble_accuracy,
    ensemble_predictions = ensemble_predictions
  )
  
  return(results)
}

# Funzione aggiornata per accuracy per gruppi
calculate_group_accuracy <- function(models, test_data, sensitive_attr, train_data, target_col = "y") {
  groups <- unique(test_data[[sensitive_attr]])
  
  results <- data.frame(
    Group = character(),
    Accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (group in groups) {
    group_data <- test_data[test_data[[sensitive_attr]] == group, ]
    true_labels <- group_data[[target_col]]
    
    # Ottieni predizioni per questo gruppo
    group_predictions <- predict_ensemble(models, group_data)
    
    # Calcola accuracy per ogni modello
    accuracies <- sapply(1:length(models), function(i) {
      mean(group_predictions[[i]] == true_labels)
    })
    
    results <- rbind(results, data.frame(
      Group = as.character(group),
      Accuracy = mean(accuracies)
    ))
  }
  
  return(results)
}