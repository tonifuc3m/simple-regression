pprepro_pred <- function(RawATP, drop_draw_size=FALSE, log_variables=FALSE, scaling=FALSE, remove_outliers=FALSE,yeo_johnson=FALSE){
  # Drop ID, names, country of origin and date columns
  drops <- c("tourney_id","p1_id", "p1_name", "p2_name", "tourney_name", 
             "p2_id", "p1_ioc", "p2_ioc" ,"tourney_date", "match_num")
  ATP <-RawATP[ , !(names(RawATP) %in% drops)]
  
  # Categorical variables to factor
  ATP$p1_win <- as.factor(ATP$p1_win)
  if (!drop_draw_size) {
    ATP$draw_size <- as.factor(ATP$drap1_size)
  }
  ATP <-ATP[ , !(names(ATP) %in% c("drap1_size"))]
  
  # Create new variables
  ATP["diff_rank_points"] <- abs(ATP["p1_rank_points"] - ATP["p2_rank_points"])
  ATP["diff_age"] <- abs(ATP["p1_age"] - ATP["p2_age"])
  ATP["diff_rank"] <- abs(ATP["p1_rank"] - ATP["p2_rank"])
  
  numeric_variables <- unlist(lapply(ATP, is.numeric))
  
  # Remove missing/uncommon values 
  ATP[ATP$p1_hand == 'U',]$p1_hand <- 'R'
  ATP[ATP$p2_hand == 'U',]$p2_hand <- 'R'
  ATP[ATP$surface == 'None',]$surface <- 'Hard'
  ATP[ATP$surface == 'Carpet',]$surface <- 'Grass'
  
  ATP$p1_hand = droplevels(ATP$p1_hand, except = c('R', 'L'))
  ATP$p2_hand = droplevels(ATP$p2_hand, except = c('R', 'L'))
  ATP$surface = droplevels(ATP$surface)
  
  # Tranform exponential variables
  if (log_variables) {
    ATP$p1_rank = log(ATP$p1_rank)
    ATP$p2_rank = log(ATP$p2_rank)
    ATP$p1_rank_points = log(ATP$p1_rank_points)
    ATP$p2_rank_points = log(ATP$p2_rank_points)
    ATP$diff_rank = log(ATP$diff_rank)
    ATP$diff_rank_points = log(ATP$diff_rank_points + 1) #Can be zero. Not to get -Inf or negative values
    if (yeo_johnson){
      YJ <- car::powerTransform(lm(ATP$diff_points ~ 1), family = "yjPower")
      # Yeo-Johnson transformation
      ATP$diff_points = car::yjPower(U = ATP$diff_points, lambda = YJ$lambda)
    }else {
      ATP$diff_points = log(ATP$diff_points + 1) #Can be zero. Not to get -Inf or negative values
    }
  }
  
  # Center & Scale
  if (scaling){
      mean_list = lapply(ATP[,numeric_variables], mean)
      sd_list = lapply(ATP[, numeric_variables], sd)
      ATP.scaled = (ATP[,numeric_variables] - mean_list ) / sd_list
  }
  else{
    ATP.scaled = ATP[,numeric_variables]
  }
  
  # Remove high leverage points
  if (remove_outliers) {
    h <- hat(ATP.scaled[,!(names(ATP.scaled) %in% c("diff_points"))])
    n <- nrow(ATP.scaled)
    p <- ncol(ATP.scaled) - 1
    hist(h, breaks = 50)
    abline(v = (qchisq(0.99, df = p) + 1) / n, col = 2)
    high_leverage_points = which(h > (qchisq(0.99, df = p) + 1) / n)
    print(sprintf("Number of high leverage points with alpha=0.99: %s (%2f%%)",
                  length(high_leverage_points),length(high_leverage_points)*100/n))
    if (length(high_leverage_points) != 0){
      ATP.scaled <- ATP.scaled[ - high_leverage_points, ]
      ATP <- ATP[ - high_leverage_points, ]
    }
  }
  
  # Re-join numeric and categorical variables
  ATP[, numeric_variables] = ATP.scaled
  #skim(ATP.scaled)
  rm(ATP.scaled)
  
  # X and Y
  X <- ATP[ , !(names(ATP) %in% c("p1_win", "diff_points"))]
  Y <- ATP[ , (names(ATP) %in% c("p1_win", "diff_points"))]
  
  returnList <- list("X" = X, "Y" = Y)
  return(returnList)
}
  
splitter_pred <- function(X, Y){
  
  # Split into train and test
  
  n = nrow(X)
  set.seed(0)
  smp_size <- floor(0.5 * n)
  train_ind <- sample(seq_len(n), size = smp_size)
  Xtrain <- X[train_ind, ]
  Xtest <- X[-train_ind, ]
  Ytrain <- Y[train_ind, ]
  Ytest <- Y[-train_ind, ]

  returnList <- list("Xtrain" = Xtrain, "Ytrain" = Ytrain, "Xtest" = Xtest, "Ytest" = Ytest)
  return(returnList)
}

PCA_pred <- function(Xtrain, Ytrain){
  
  # Perform PCA of the nummeric variables
  numeric_variables <- unlist(lapply(Xtrain, is.numeric))
  
  Xtrain.numm <- subset(Xtrain, select = numeric_variables)
  pcaXtrain.numm <- princomp(x = Xtrain.numm, cor = TRUE)
  
  # Join PCs with categorical variables and output variable
  factor_variables <- unlist(lapply(Xtrain, is.factor))
  train.PCA = cbind(Xtrain[, factor_variables], 
                   data.frame(pcaXtrain.numm$scores),
                   data.frame(Ytrain$diff_points))
  
  returnList = list("train.PCA_object" = pcaXtrain.numm, "train.PCA" = train.PCA)
  return(returnList)
}

stepwise_pred <- function(train.PCA, BIC=TRUE, interactions=FALSE){
  
  # Perform stepwise regression
  if (interactions) {
    modPCA <- lm(Ytrain.diff_points ~ .^2, data = train.PCA)
  } else{
    modPCA <- lm(Ytrain.diff_points ~ ., data = train.PCA)
  }
 
  # Select BIC or AIC criteria
  if (BIC) {
    k = log(nrow(train.PCA))
  }
  else {
    k = 2
  }
  
  modPCAAIC <- MASS::stepAIC(modPCA, k = 2 ,trace = 0, direction = "both")

  predictors.PCA <- unlist(names(modPCAAIC$coefficients)[2:length(modPCAAIC$coefficients)])
  
  returnList = list("model.PCA.AIC" = modPCAAIC, "predictors.PCA"= predictors.PCA)
  
  return(returnList)
}

refit_pred <- function(Xtest, Ytest, predictors, model, isPCA){
  
  if (isPCA == 1){
    numeric_variables <- unlist(lapply(Xtest, is.numeric))
    factor_variables <- unlist(lapply(Xtest, is.factor))
    
    # Transform numeric test data
    Xtest.PCA = data.frame(predict(model, 
                                   newdata = Xtest[,numeric_variables]))
    
    # Dummify tourney_level variable
    aux = data.frame(model.matrix(~Xtest$tourney_level))
    tl = aux[,2:length(aux)]
    names(tl) <- substring(names(tl), 7)
    
    # Join PCs with categorical variables and output variable
    test.PCA = cbind(Xtest[, factor_variables], 
                     tl,
                     data.frame(Xtest.PCA))
    test.PCA$tourney_level <- NULL
    test.PCA <- test.PCA[ , (names(test.PCA) %in% predictors)]
    
    test = cbind(test.PCA, data.frame(Ytest$diff_points))
    
  }else{
    test = cbind(Xtest[, (names(Xtest) %in% predictors)],
                 data.frame(Ytest$diff_points))
  }

  
  # Fit model with the selected predictors
  model.final <- lm(Ytest.diff_points ~ ., data = test)
  
  return(model.final)
}

model_diagnostics <- function(mod){
  
  # Normality
  st <- shapiro.test(mod$residuals)
  if (st$p.value < 0.05){
    print("Normality does not hold according to Shapiro test")
  } else {
    print("Normality holds according to Shapiro test")
  }
  
  # Homoscedasticity
  bpt <- car::ncvTest(mod)
  if (bpt$p < 0.05){
    print("Homoscedasticity does not hold according to Breusch–Pagan test.")
  } else {
    print("Homoscedasticity holds according to Breusch–Pagan test.")
  }
  
  # Independence
  dbt = car::durbinWatsonTest(mod)
  if (dbt$p < 0.05){
    print("Independence does not hold according to Durbin–Watson test.")
  } else{
    print("Independence holds according to Durbin–Watson test.")
  }
  
  #Plots
  par(mfrow=c(2,2)) 
  layout(matrix(c(1,2,3,3), 2, 2, byrow = TRUE))
  #par(mfrow=c(1,2)) 
  plot(mod,1)
  plot(mod, 2)
  plot(mod$residuals, type = "o")
}


val_assump_pred <- function(mod, X){
  
  numeric_variables <- unlist(lapply(X, is.numeric))
  X = X[, numeric_variables]
  
  # Linearity
  cat("\n\n\n####### LINEARITY TEST ######\n")
  plot(mod, 1) 
  print("We expect to see NO trend")
  readline(prompt="Press [enter] to continue")
  print("\n\n\n")
  
  
  
  # Normality
  cat("\n\n\n####### NORMALITY TEST ######\n")
  plot(mod, 2)
  print("We expect to see points along diagonal line")
  st <- shapiro.test(mod$residuals)
  if (st$p.value < 0.05){
    print("Normality does not hold according to Shapiro test")
  }
  readline(prompt="Press [enter] to continue")

  
  # Homoscedasticity
  cat("\n\n\n####### HOMOSCEDASTICITY TEST ######\n")
  plot(mod, 3) 
  print("We expect to see no trend")
  bpt <- car::ncvTest(mod)
  if (bpt$p < 0.05){
    print("Homoscedasticity does not hold according to Breusch–Pagan test.")
    print("Caution, not definitive result, check scale-location plot!")
  }
  readline(prompt="Press [enter] to continue")
  
  
  # Independence
  cat("\n\n\n####### INDEPENDENCE TEST ######\n")
  plot(mod$residuals, type = "o") 
  print("We expect the series to show no tracking of the residuals.")
  print("Tracking is associated to positive autocorrelation")
  dbt = car::durbinWatsonTest(mod)
  if (dbt$p < 0.05){
    print("Independenc does not hold according to Durbin–Watson test .")
  }
  readline(prompt="Press [enter] to continue")
  
  
  # Multicollinearity
  cat("\n\n\n####### MULTICOLLINEARITY TEST ######\n")
  print(car::vif(mod))
  print("If values greater than 5, we have a problem with multicollinearity")
  readline(prompt="Press [enter] to continue")
  
  
  # Outliers
  cat("\n\n\n####### OUTLIERS TEST ######\n")
  plot(modPCA, 5)
  print("Check for points farther away than Cook's distance")
  readline(prompt="Press [enter] to continue")
  
  
  # High-leverage points
  cat("\n\n\n####### HIGH-LEVERAGE POINTS TEST ######\n")
  h <- hat(X)
  n <- length(X)
  p <- 21
  hist(h, breaks = 20)
  abline(v = (qchisq(0.90, df = p) + 1) / n, col = 2)
  high_leverage_points = which(h > (qchisq(0.90, df = p) + 1) / n) # ninguno????
  print(high_leverage_points)
}

asses_assump_pred <- function(list_failed_assumptions){
  
  return()
}
  