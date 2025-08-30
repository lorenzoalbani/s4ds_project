calculate_fairness_measures <- function(uncertainties, sensitive_attr) {
  groups <- unique(sensitive_attr)
  G0 <- groups[1]  # minority group
  G1 <- groups[2]  # majority group
  
  # Group-level uncertainties (media per gruppo)
  U_G0 <- mean(uncertainties[sensitive_attr == G0])
  U_G1 <- mean(uncertainties[sensitive_attr == G1])
  
  # Fairness measure F_u = U(G=0) / U(G=1) - Eq. 27
  fairness_ratio <- U_G0 / U_G1
  
  return(list(
    U_G0 = U_G0,
    U_G1 = U_G1,
    fairness_ratio = fairness_ratio,
    is_fair = abs(fairness_ratio - 1) <= 0.2  # threshold dal paper
  ))
}

calculate_uncertainty_fairness <- function(uncertainties, sensitive_attr_values) {
  # Estrai le incertezze
  epistemic <- uncertainties$epistemic
  aleatoric <- uncertainties$aleatoric
  predictive <- uncertainties$predictive
  
  # Gruppi
  group_0 <- sensitive_attr_values == 0
  group_1 <- sensitive_attr_values == 1
  
  # Calcola le medie per ciascun gruppo
  mean_epistemic_0 <- mean(epistemic[group_0])
  mean_epistemic_1 <- mean(epistemic[group_1])
  
  mean_aleatoric_0 <- mean(aleatoric[group_0])
  mean_aleatoric_1 <- mean(aleatoric[group_1])
  
  mean_predictive_0 <- mean(predictive[group_0])
  mean_predictive_1 <- mean(predictive[group_1])
  
  # Calcola le misure di fairness (rapporto tra gruppi)
  F_Epis <- mean_epistemic_0 / mean_epistemic_1
  F_Alea <- mean_aleatoric_0 / mean_aleatoric_1
  F_Pred <- mean_predictive_0 / mean_predictive_1
  
  return(data.frame(
    F_Epis = round(F_Epis, 2),
    F_Alea = round(F_Alea, 2),
    F_Pred = round(F_Pred, 2)
  ))
}
