
# Load necessary libraries
library(readr)

# Read the data
adult_data <- read_csv('data/real_datasets/adult/adult.data', col_names = FALSE)
adult_test <- read_csv('data/real_datasets/adult/adult.test', skip = 1, col_names = FALSE)

# Assign column names
colnames(adult_data) <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income")
colnames(adult_test) <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income")

# View summary
summary(adult_data)
summary(adult_test)
