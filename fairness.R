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