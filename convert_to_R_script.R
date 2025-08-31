
# Caricamento del dataset Adult in R
adult_data <- read.csv("adult.data", header = FALSE, sep = ",", strip.white = TRUE, na.strings = "?")

# Assegnazione dei nomi delle colonne
colnames(adult_data) <- c(
  "age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "sex",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
)

# Visualizzazione delle prime righe
head(adult_data)

# Pulizia dei dati
adult_data <- na.omit(adult_data)
adult_data$income <- factor(adult_data$income, levels = c("<=50K", ">50K"))
