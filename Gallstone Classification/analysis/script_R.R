# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Daftar package yang diperlukan
packages <- c("mlbench", "caret", "pROC", "ROCR", "class", "nnet", 
              "ggplot2", "cluster", "factoextra", "clusterSim", 
              "reshape2", "rpart", "rpart.plot", "MASS", "e1071")

# Cek dan install package yang belum ada
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, dependencies = TRUE)
}

# Load semua packages dengan error handling
for(pkg in packages) {
  if(!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

cat("All packages loaded successfully!\n")

# Set CRAN mirror jika belum di-set
if(length(getOption("repos")) == 0 || getOption("repos")["CRAN"] == "@CRAN@") {
  options(repos = c(CRAN = "https://cloud.r-project.org/"))
}

# Load library yang diperlukan (dengan error handling)
required_libs <- c("mlbench", "caret", "pROC", "ROCR", "class", 
                   "nnet", "ggplot2", "rpart", "rpart.plot", "MASS", "readxl")

for(lib in required_libs) {
  if(!require(lib, character.only = TRUE, quietly = TRUE)) {
    install.packages(lib, dependencies = TRUE)
    library(lib, character.only = TRUE)
  }
}

# Load dataset diabetes
df <- read_excel("dataset-gallstone.xlsx")
head(df)

str(df)


# Cek distribusi target
table(df$`Gallstone Status`)
prop.table(table(df$`Gallstone Status`))

# Ubah dari numeric ke faktor, dengan label yang benar
df$`Gallstone Status` <- factor(df$`Gallstone Status`,
                                levels = c(0, 1),
                                labels = c("neg", "pos"))

#split data 80% training, 20% testing
set.seed(123)
train_index <- createDataPartition(df$`Gallstone Status`, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

cat("Jumlah data train:", nrow(train_data), "\n")
cat("Jumlah data test:", nrow(test_data), "\n")

cat("\nDistribusi kelas di train data:\n")
print(table(train_data$`Gallstone Status`))

cat("\nDistribusi kelas di test data:\n")
print(table(test_data$`Gallstone Status`))



### --- LOGISTIC REGRESSION --- ###
# Training dan Prediksi
# Model Regresi Logistik
model_logit <- glm(`Gallstone Status` ~ ., data = train_data, family = binomial)
summary(model_logit)

# Prediksi pada data test
pred_prob_logit <- predict(model_logit, newdata = test_data, type = "response")
pred_class_logit <- ifelse(pred_prob_logit > 0.5, "pos", "neg")
pred_class_logit <- factor(pred_class_logit, levels = c("neg", "pos"))
pred_class_logit
levels(test_data$`Gallstone Status`) <- c("neg", "pos")

# Evaluasi Logistic Regression
# Confusion Matrix
conf_logit <- confusionMatrix(pred_class_logit, test_data$`Gallstone Status`, positive = "pos")
print(conf_logit)

# ROC Curve
roc_logit <- roc(test_data$`Gallstone Status`, pred_prob_logit)
plot(roc_logit, main = "ROC Curve - Logistic Regression",
     col = "blue", lwd = 2, print.auc = TRUE)
abline(a = 0, b = 1, lty = 2, col = "red")




### --- MODEL DECISION TREE --- ###
# Training Decision Tree
# Model Decision Tree
model_tree <- rpart(`Gallstone Status` ~ ., data = train_data, 
                    method = "class",
                    control = rpart.control(cp = 0.01, minsplit = 20))

# Visualisasi Tree
rpart.plot(model_tree, 
           type = 4, 
           extra = 101,
           box.palette = "RdYlGn",
           shadow.col = "gray",
           main = "Decision Tree untuk Prediksi Gallstone Status")

# Print tree rules
print(model_tree)


## Prediksi dan Evaluasi Decision Tree
# Prediksi
pred_prob_tree <- predict(model_tree, newdata = test_data, type = "prob")[,2]
pred_class_tree <- predict(model_tree, newdata = test_data, type = "class")

# Confusion Matrix
conf_tree <- confusionMatrix(pred_class_tree, test_data$`Gallstone Status`, positive = "pos")
print(conf_tree)

# ROC Curve
roc_tree <- roc(test_data$`Gallstone Status`, pred_prob_tree)
plot(roc_tree, main = "ROC Curve - Decision Tree",
     col = "darkgreen", lwd = 2, print.auc = TRUE)
abline(a = 0, b = 1, lty = 2, col = "red")


## Feature Importance dari Decision Tree
# Variable Importance
importance <- model_tree$variable.importance
importance_df <- data.frame(
  Feature = names(importance),
  Importance = importance
)
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

# Plot importance
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance - Decision Tree",
       x = "Features",
       y = "Importance Score") +
  theme_minimal()

print(importance_df)


## Pruning Decision Tree
# Cek complexity parameter
printcp(model_tree)
plotcp(model_tree)

# Pruning tree dengan cp optimal
cp_optimal <- model_tree$cptable[which.min(model_tree$cptable[,"xerror"]),"CP"]
cat("Optimal CP:", cp_optimal, "\n")

# Pruned tree
model_tree_pruned <- prune(model_tree, cp = cp_optimal)

# Visualisasi pruned tree
rpart.plot(model_tree_pruned, 
           type = 4, 
           extra = 101,
           box.palette = "RdYlGn",
           main = "Pruned Decision Tree")

# Evaluasi pruned tree
pred_class_tree_pruned <- predict(model_tree_pruned, newdata = test_data, type = "class")
conf_tree_pruned <- confusionMatrix(pred_class_tree_pruned, test_data$`Gallstone Status`, positive = "pos")
print(conf_tree_pruned)




### --- Model 3: K-Nearest Neighbors (KNN) --- ###
## Preprocessing untuk KNN
# KNN memerlukan normalisasi data
# Pisahkan features dan target
train_features <- train_data[, -1]  # kolom 9 adalah diabetes
test_features <- test_data[, -1]
train_target <- train_data$`Gallstone Status`
test_target <- test_data$`Gallstone Status`

# Normalisasi menggunakan scale
train_scaled <- as.data.frame(scale(train_features))
test_scaled <- as.data.frame(scale(test_features))

# Cek hasil normalisasi
summary(train_scaled)


## Training KNN dengan Different K Values
# Test multiple k values
k_values <- c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
accuracy_k <- numeric(length(k_values))

for(i in 1:length(k_values)) {
  k <- k_values[i]
  pred_knn <- knn(train = train_scaled, 
                  test = test_scaled, 
                  cl = train_target, 
                  k = k,
                  prob = TRUE)
  
  cm <- confusionMatrix(pred_knn, test_target)
  accuracy_k[i] <- cm$overall['Accuracy']
}

# Plot accuracy vs k
plot_df <- data.frame(K = k_values, Accuracy = accuracy_k)
ggplot(plot_df, aes(x = K, y = Accuracy)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  geom_vline(xintercept = k_values[which.max(accuracy_k)], 
             linetype = "dashed", color = "darkgreen") +
  labs(title = "KNN: Accuracy vs K Value",
       subtitle = paste("Optimal K =", k_values[which.max(accuracy_k)]),
       x = "K (Number of Neighbors)",
       y = "Accuracy") +
  theme_minimal()

cat("Optimal K:", k_values[which.max(accuracy_k)], 
    "with Accuracy:", max(accuracy_k), "\n")


## KNN dengan Optimal K
# Gunakan optimal k
optimal_k <- k_values[which.max(accuracy_k)]

# Prediksi dengan optimal k
pred_knn_optimal <- knn(train = train_scaled, 
                        test = test_scaled, 
                        cl = train_target, 
                        k = optimal_k,
                        prob = TRUE)

# Get probabilities
prob_knn <- attr(pred_knn_optimal, "prob")
# Adjust probability for class
pred_prob_knn <- ifelse(pred_knn_optimal == "pos", prob_knn, 1 - prob_knn)

# Confusion Matrix
conf_knn <- confusionMatrix(pred_knn_optimal, test_target, positive = "pos")
print(conf_knn)

# ROC Curve
roc_knn <- roc(test_target, pred_prob_knn)
plot(roc_knn, main = paste("ROC Curve - KNN (k =", optimal_k, ")"),
     col = "purple", lwd = 2, print.auc = TRUE)
abline(a = 0, b = 1, lty = 2, col = "red")





### --- Model 4: Linear Discriminant Analysis (LDA) --- ###
## Training dan Evaluasi Model LDA
# Model LDA
model_lda <- lda(`Gallstone Status` ~ ., data = train_data)
print(model_lda)

# Prediksi
pred_lda <- predict(model_lda, newdata = test_data)
pred_class_lda <- pred_lda$class
pred_prob_lda <- pred_lda$posterior[, 2]

# Confusion Matrix
conf_lda <- confusionMatrix(pred_class_lda, test_data$`Gallstone Status`, positive = "pos")
print(conf_lda)

# ROC Curve
roc_lda <- roc(test_data$`Gallstone Status`, pred_prob_lda)
plot(roc_lda, main = "ROC Curve - LDA",
     col = "orange", lwd = 2, print.auc = TRUE)
abline(a = 0, b = 1, lty = 2, col = "red")




### --- Eksplorasi Data --- ###
# 1) Load packages (install bila perlu)
pkgs <- c("readxl", "tidyverse", "skimr", "naniar", "GGally", "psych", "janitor")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

suppressPackageStartupMessages({
  library(readxl)
  library(tidyverse)   # termasuk dplyr, ggplot2, tidyr, %>%
  library(skimr)      # ringkasan cepat
  library(naniar)     # visualisasi missing
  library(GGally)     # ggpairs
  library(psych)      # describe
  library(janitor)    # clean_names, tabyl
})

# 2) Baca worksheet (pilih sheet pertama kalau banyak)
sheets <- excel_sheets("dataset-gallstone.xlsx")
message("Sheet tersedia: ", paste(sheets, collapse = ", "))
df <- read_excel("dataset-gallstone.xlsx", sheet = sheets[1]) %>% 
  clean_names()    # ubah nama kolom jadi snake_case aman

# 3) Simpan folder output (grafik & ringkasan)
out_dir <- "EDA_output"
if (!dir.exists(out_dir)) dir.create(out_dir)

# 4) Struktur & preview
glimpse(df)           # struktur cepat
head(df, 10)
tail(df, 10)

# 5) Ringkasan cepat (skimr)
skim_out <- skim(df)
print(skim_out)
# juga simpan ringkasan menjadi csv
write.csv(as.data.frame(skim_out), file = file.path(out_dir, "skim_summary.csv"), row.names = FALSE)

# 6) Informasi missing
# Persentase missing per variabel
missing_pct <- df %>% summarise(across(everything(), ~ mean(is.na(.)) * 100)) %>% 
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>% 
  arrange(desc(pct_missing))
print(missing_pct)
write.csv(missing_pct, file = file.path(out_dir, "missing_percentage.csv"), row.names = FALSE)

# Visualisasi missing overall
png(file.path(out_dir, "missing_map.png"), width = 1000, height = 500)
vis_miss(df) + ggtitle("Missingness map")
dev.off()

# 7) Tipe variabel & jumlah unique
var_types <- map_chr(df, ~ class(.x)[1])
unique_counts <- map_int(df, ~ n_distinct(.x, na.rm = TRUE))
var_info <- tibble(variable = names(df), type = var_types, unique = unique_counts)
print(var_info)
write.csv(var_info, file = file.path(out_dir, "variable_info.csv"), row.names = FALSE)

# 8) Pisahkan numerik dan kategorikal
num_vars <- df %>% select(where(is.numeric))
cat_vars <- df %>% select(where(~ !is.numeric(.)))

# 9) Statistik deskriptif numerik (psych::describe + summary)
if(ncol(num_vars) > 0) {
  desc_num <- psych::describe(num_vars)
  print(desc_num)
  write.csv(as.data.frame(desc_num), file = file.path(out_dir, "numeric_describe.csv"))
}

# 10) Korelasi (hanya numeric)
if(ncol(num_vars) > 1) {
  cor_mat <- cor(num_vars, use = "pairwise.complete.obs")
  write.csv(cor_mat, file = file.path(out_dir, "correlation_matrix.csv"))
  # heatmap korelasi sederhana
  png(file.path(out_dir, "correlation_heatmap.png"), width = 900, height = 700)
  corrplot::corrplot(cor_mat, method = "color", tl.cex = 0.8, number.cex = 0.7)
  dev.off()
}

# 11) Frekuensi untuk kategorikal
if(ncol(cat_vars) > 0) {
  freq_list <- map(cat_vars, ~ janitor::tabyl(.x) %>% arrange(desc(n)))
  # simpan masing-masing sebagai csv
  for(nm in names(freq_list)) {
    write.csv(freq_list[[nm]], file = file.path(out_dir, paste0("freq_", nm, ".csv")), row.names = FALSE)
  }
}

# 12) Visualisasi variabel penting:
# a) Histogram & density untuk semua numeric
for(nm in names(num_vars)) {
  p <- ggplot(df, aes(x = .data[[nm]])) +
    geom_histogram(bins = 30, na.rm = TRUE) +
    geom_density(aes(y = ..count..), alpha = 0.2, na.rm = TRUE) +
    labs(title = paste("Histogram:", nm), x = nm) +
    theme_minimal()
  ggsave(filename = file.path(out_dir, paste0("hist_", nm, ".png")), plot = p, width = 7, height = 4)
}

# b) Boxplot numeric vs kategori (untuk tiap kategori pertama jika ada kategori)
if(ncol(cat_vars) > 0 & ncol(num_vars) > 0) {
  first_cat <- names(cat_vars)[1]
  for(nm in names(num_vars)) {
    p <- ggplot(df, aes_string(x = first_cat, y = nm)) +
      geom_boxplot(na.rm = TRUE) +
      labs(title = paste("Boxplot:", nm, "by", first_cat)) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ggsave(filename = file.path(out_dir, paste0("box_", nm, "_by_", first_cat, ".png")), plot = p, width = 8, height = 4.5)
  }
}

# c) Scatterplot matrix (ggpairs) dari maximal 10 numeric untuk performa
if(ncol(num_vars) > 1) {
  use_vars <- names(num_vars)[1:min(10, ncol(num_vars))]
  png(file.path(out_dir, "scatter_matrix.png"), width = 1500, height = 1500)
  GGally::ggpairs(df %>% select(all_of(use_vars)), progress = FALSE)
  dev.off()
}

# 13) Deteksi outlier sederhana (IQR) untuk setiap numeric
outlier_report <- map_df(names(num_vars), function(nm) {
  x <- num_vars[[nm]]
  q <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  lower <- q[1] - 1.5 * iqr
  upper <- q[2] + 1.5 * iqr
  tibble(variable = nm,
         n_obs = sum(!is.na(x)),
         n_outliers = sum(x < lower | x > upper, na.rm = TRUE),
         pct_outliers = mean(x < lower | x > upper, na.rm = TRUE) * 100)
})
write.csv(outlier_report, file = file.path(out_dir, "outlier_report.csv"), row.names = FALSE)
print(outlier_report)

# 14) Correlation with target (jika ada kolom bernama 'target' atau 'y')
target_names <- c("target", "y", "label")
found_target <- intersect(names(df), target_names)
if(length(found_target) > 0 & ncol(num_vars) > 0) {
  tar <- found_target[1]
  cor_with_target <- cor(num_vars, df[[tar]], use = "pairwise.complete.obs")
  cor_df <- tibble(variable = names(cor_with_target), correlation_with_target = cor_with_target)
  write.csv(cor_df, file = file.path(out_dir, "cor_with_target.csv"), row.names = FALSE)
  print(cor_df %>% arrange(desc(abs(correlation_with_target))))
}

# 15) Save cleaned (opsional): convert string kosong ke NA dan simpan sampel
df_cleaned <- df %>%
  mutate(across(where(is.character), ~ na_if(trimws(.), "")))

write.csv(df_cleaned, file = file.path(out_dir, "data_cleaned_sample.csv"), row.names = FALSE)

# 16) Ringkasan akhir untuk user
message("EDA selesai. File ringkasan dan grafik disimpan di folder: ", normalizePath(out_dir))

View(df)






