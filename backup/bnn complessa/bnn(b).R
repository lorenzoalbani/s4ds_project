library(torch)
library(torchvision)

# ========================================
# 1. LAYER VARIAZIONALE BAYESIANO
# ========================================

# Classe per layer lineare bayesiano
BayesianLinear <- nn_module(
  "BayesianLinear",
  
  initialize = function(in_features, out_features, prior_sigma1 = 0, prior_sigma2 = 6, prior_pi = 0.5) {
    # Parametri del layer
    self$in_features <- in_features
    self$out_features <- out_features
    
    # Prior mixture parameters
    self$prior_sigma1 <- prior_sigma1
    self$prior_sigma2 <- prior_sigma2
    self$prior_pi <- prior_pi
    
    # Parametri variazionali per i pesi (mean e log_variance)
    # Inizializzazione: mean ~ N(0,1), log_var inizializzato per dare variance ragionevole
    self$weight_mu <- nn_parameter(torch_randn(out_features, in_features))
    self$weight_logvar <- nn_parameter(torch_full(c(out_features, in_features), -5))  # piccola variance iniziale
    
    # Parametri variazionali per i bias
    self$bias_mu <- nn_parameter(torch_randn(out_features))
    self$bias_logvar <- nn_parameter(torch_full(c(out_features), -5))
  },
  
  # Campionamento dei pesi usando reparameterization trick
  sample_weights = function() {
    # Sample weights: mu + sigma * epsilon
    weight_eps <- torch_randn_like(self$weight_mu)
    weight_sigma <- torch_exp(0.5 * self$weight_logvar)
    weights <- self$weight_mu + weight_sigma * weight_eps
    
    # Sample bias
    bias_eps <- torch_randn_like(self$bias_mu)  
    bias_sigma <- torch_exp(0.5 * self$bias_logvar)
    bias <- self$bias_mu + bias_sigma * bias_eps
    
    return(list(weights = weights, bias = bias))
  },
  
  # Forward pass
  forward = function(x) {
    sampled <- self$sample_weights()
    # torch_linear non esiste in R, usa torch_addmm
    return(torch_addmm(sampled$bias, x, sampled$weights$t()))
  },
  
  # Calcola KL divergence per questo layer
  kl_divergence = function() {
    # KL tra posteriore variazionale e prior mixture
    # Questo è il pezzo più complesso - approssimazione
    
    # Per semplicità, qui uso una approssimazione
    # In realtà dovresti implementare la KL esatta con il mixture prior
    weight_var <- torch_exp(self$weight_logvar)
    bias_var <- torch_exp(self$bias_logvar)
    
    # KL approssimato (da migliorare con mixture exact)
    kl_weight <- 0.5 * torch_sum(self$weight_mu^2 + weight_var - self$weight_logvar - 1)
    kl_bias <- 0.5 * torch_sum(self$bias_mu^2 + bias_var - self$bias_logvar - 1)
    
    return(kl_weight + kl_bias)
  }
)

# ========================================
# 2. RETE NEURALE BAYESIANA COMPLETA
# ========================================

BayesianNN <- nn_module(
  "BayesianNN",
  
  initialize = function(input_size, hidden_sizes, output_size, prior_params = list()) {
    # Parametri prior
    prior_sigma1 <- prior_params$sigma1 %||% 0
    prior_sigma2 <- prior_params$sigma2 %||% 6
    prior_pi <- prior_params$pi %||% 0.5
    
    # Costruisci layers
    layers <- list()
    
    # Input layer
    layers[[1]] <- BayesianLinear(input_size, hidden_sizes[1], prior_sigma1, prior_sigma2, prior_pi)
    
    # Hidden layers
    if (length(hidden_sizes) > 1) {
      for (i in 2:length(hidden_sizes)) {
        layers[[i]] <- BayesianLinear(hidden_sizes[i-1], hidden_sizes[i], prior_sigma1, prior_sigma2, prior_pi)
      }
    }
    
    # Output layer
    layers[[length(layers) + 1]] <- BayesianLinear(tail(hidden_sizes, 1), output_size, prior_sigma1, prior_sigma2, prior_pi)
    
    self$layers <- nn_module_list(layers)
    self$activation <- nn_relu()
  },
  
  forward = function(x) {
    # Forward pass attraverso tutti i layers
    for (i in 1:(length(self$layers) - 1)) {
      x <- self$activation(self$layers[[i]](x))
    }
    # Output layer senza attivazione (per classificazione usiamo softmax dopo)
    x <- self$layers[[length(self$layers)]](x)
    return(x)
  },
  
  # Calcola KL divergence totale della rete
  total_kl_divergence = function() {
    total_kl <- 0
    for (i in 1:length(self$layers)) {
      total_kl <- total_kl + self$layers[[i]]$kl_divergence()
    }
    return(total_kl)
  },
  
  # Predict con uncertainty quantification
  predict_with_uncertainty = function(x, n_samples = 10) {
    predictions <- list()
    
    # Genera n_samples predizioni
    for (i in 1:n_samples) {
      with_no_grad({
        pred <- nnf_softmax(self$forward(x), dim = 2)
        predictions[[i]] <- pred
      })
    }
    
    # Stack e calcola statistiche
    all_preds <- torch_stack(predictions, dim = 1)  # [n_samples, batch_size, n_classes]
    
    mean_pred <- torch_mean(all_preds, dim = 1)      # predizione media
    std_pred <- torch_std(all_preds, dim = 1)        # incertezza
    
    return(list(mean = mean_pred, std = std_pred, samples = all_preds))
  }
)

# ========================================
# 3. LOSS FUNCTION PERSONALIZZATA
# ========================================

BayesianLoss <- function(model, outputs, targets, lambda = 2000, n_batches = 1) {
  # Classification loss (cross-entropy)
  classification_loss <- nn_cross_entropy_loss()(outputs, targets)
  
  # KL divergence
  kl_loss <- model$total_kl_divergence()
  
  # Total loss secondo formula (22)
  # Nota: kl_loss va diviso per n_batches per scalare correttamente
  total_loss <- classification_loss + (lambda / n_batches) * kl_loss
  
  return(list(
    total = total_loss,
    classification = classification_loss,
    kl = kl_loss
  ))
}

# ========================================
# 4. TRAINING LOOP ESEMPIO
# ========================================

train_bayesian_nn <- function(model, train_loader, val_loader = NULL, epochs = 100, lr = 0.001, lambda = 2000) {
  
  # Optimizer Adam
  optimizer <- optim_adam(model$parameters, lr = lr)
  
  # Early stopping parameters
  best_val_loss <- Inf
  patience <- 10
  patience_counter <- 0
  
  n_batches <- length(train_loader)
  
  for (epoch in 1:epochs) {
    model$train()
    epoch_loss <- 0
    epoch_class_loss <- 0
    epoch_kl_loss <- 0
    
    # Training
    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()
      
      # Forward pass
      outputs <- model(batch$x)
      
      # Compute loss
      losses <- BayesianLoss(model, outputs, batch$y, lambda, n_batches)
      
      # Backward pass
      losses$total$backward()
      optimizer$step()
      
      # Accumulate losses
      epoch_loss <- epoch_loss + as.numeric(losses$total)
      epoch_class_loss <- epoch_class_loss + as.numeric(losses$classification)
      epoch_kl_loss <- epoch_kl_loss + as.numeric(losses$kl)
    })
    
    # Validation
    if (!is.null(val_loader)) {
      model$eval()
      val_loss <- 0
      val_acc <- 0
      n_val_batches <- 0
      
      with_no_grad({
        coro::loop(for (batch in val_loader) {
          outputs <- model(batch$x)
          losses <- BayesianLoss(model, outputs, batch$y, lambda, n_batches)
          val_loss <- val_loss + as.numeric(losses$total)
          
          # Accuracy
          preds <- torch_argmax(nnf_softmax(outputs, dim = 2), dim = 2)
          acc <- torch_mean((preds == batch$y)$to(dtype = torch_float()))
          val_acc <- val_acc + as.numeric(acc)
          n_val_batches <- n_val_batches + 1
        })
      })
      
      val_loss <- val_loss / n_val_batches
      val_acc <- val_acc / n_val_batches
      
      # Early stopping
      if (val_loss < best_val_loss) {
        best_val_loss <- val_loss
        patience_counter <- 0
        # Salva il miglior modello qui se necessario
      } else {
        patience_counter <- patience_counter + 1
        if (patience_counter >= patience) {
          cat("Early stopping at epoch", epoch, "\n")
          break
        }
      }
      
      cat(sprintf("Epoch %d: Loss=%.4f, Class=%.4f, KL=%.4f, Val_Loss=%.4f, Val_Acc=%.4f\n", 
                  epoch, epoch_loss/n_batches, epoch_class_loss/n_batches, 
                  epoch_kl_loss/n_batches, val_loss, val_acc))
    } else {
      cat(sprintf("Epoch %d: Loss=%.4f, Class=%.4f, KL=%.4f\n", 
                  epoch, epoch_loss/n_batches, epoch_class_loss/n_batches, epoch_kl_loss/n_batches))
    }
  }
  
  return(model)
}

