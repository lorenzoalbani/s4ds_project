build_bnn_model <- function(input_shape, dropout_rate = 0.2) {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = input_shape) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  return(model)
}

predict_with_uncertainty <- function(model, x_test, n_samples = 50) {
  preds <- replicate(n_samples, {
    # Dropout attivo anche in fase di predizione
    k_set_learning_phase(1)
    predict(model, x_test)
  }, simplify = "array")
  
  # Calcolo delle incertezze
  mean_preds <- apply(preds, 1, mean)
  epistemic_uncertainty <- apply(preds, 1, var)
  aleatoric_uncertainty <- mean_preds * (1 - mean_preds)
  
  return(list(mean = mean_preds,
              epistemic = epistemic_uncertainty,
              aleatoric = aleatoric_uncertainty))
}

