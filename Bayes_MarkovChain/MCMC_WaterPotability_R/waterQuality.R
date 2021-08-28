# Data was obtained from https://www.kaggle.com/adityakadiwal/water-potability in June 2021.

# setwd("")   # your working directory where your csv file from kaggle is located (see line 1 for url to download csv file)


library(R2jags)
library(rjags)
library(coda)
library(lattice)


##### Read data
water <- read.csv("water_potability.csv")    # csv can be downloaded from kaggle (see line 1 for url)
water <- subset(water, select=c(10,1,2,4,5,6,7,8,9))
dim(water)


##### Show columns that contain NAs (2 possibilities)
colnames(water)[colSums(is.na(water)) > 0]
# names(which(colSums(is.na(water)) > 0))


##### Assign data
potability <- water$Potability
pH <- water$ph
hardness <- water$Hardness
chloramines <- water$Chloramines
sulfate <- water$Sulfate
conductivity <- water$Conductivity
organic_carbon <- water$Organic_carbon
trihalomethanes <- water$Trihalomethanes
turbidity <- water$Turbidity
N <- length(potability)
beta.mu <- c(0,0,0,0,0,0,0,0,0)
beta.prec <- diag(9) * 0.01


##### Evaluate mean and sd values for the columns that contain NAs
pH_vals <- c(mean(na.omit(pH)), sd(na.omit(pH)))
sulfate_vals <- c(mean(na.omit(sulfate)), sd(na.omit(sulfate)))
trihalomethanes_vals <- c(mean(na.omit(trihalomethanes)), sd(na.omit(trihalomethanes)))


##### Assign parameters for the model
to.dat <- list("N", "potability", "pH", "hardness", "chloramines", "sulfate", 
               "conductivity", "organic_carbon", "trihalomethanes", "turbidity",
               "pH_vals", "sulfate_vals", "trihalomethanes_vals", 
               "beta.mu", "beta.prec")


##### Define model
mod <- function(){
  
  ## Sampling density
  for (i in 1:N){
    potability[i] ~ dbern(p[i])   # Bernouille distribution --> binary outcome
    logit(p[i]) <- mu[i]          # logit link
    mu[i] <-  beta[1]
            + beta[2] * pH[i]
            + beta[3] * hardness[i]
            + beta[4] * chloramines[i]
            + beta[5] * sulfate[i]
            + beta[6] * conductivity[i]
            + beta[7] * organic_carbon[i]
            + beta[8] * trihalomethanes[i]
            + beta[9] * turbidity[i]
    
    pH[i] ~ dnorm(pH_vals[1], pH_vals[2])
    sulfate[i] ~ dnorm(sulfate_vals[1], sulfate_vals[2])
    trihalomethanes[i] ~ dnorm(trihalomethanes_vals[1], trihalomethanes_vals[2])
  }
  ## Prior density
  beta[1:9] ~ dmnorm(beta.mu, beta.prec)
}


##### Define starting values and params to be monitored for JAGS
inits1 <- list("b"=c(1,1,1,1,1,1,1,1,1))
inits2 <- list("b"=c(-1,-1,-1,-1,-1,-1,-1,-1,-1))
inits_list <- list(inits1, inits2)
params <- c("beta")
# params <- c("beta", "p")


##### Posterior (Run sampler)
set.seed(121)
turnout.fit <- jags(data=to.dat, inits=inits_list, parameters.to.save=params, model.file=mod, n.chains=2, n.iter=10000, n.burnin=1000)
print(turnout.fit)
# summary(turnout.fit)


##### Post Processing (Graphical Visualization)
traceplot(turnout.fit)
xyplot(as.mcmc(turnout.fit))
# plot(turnout.fit)
# plot(as.mcmc(turnout.fit))


##### Summarize Bayesin MCMC Output if "p" is included in params (line 82)
turnout.mcmc <- as.mcmc(turnout.fit)
# turnout.mat <- as.matrix(turnout.mcmc)
# turnout.out <- as.data.frame(turnout.mat)
# p <- turnout.out[, grep("p[", colnames(turnout.out), fixed = T)]
# p.mean <- apply(p, 2, mean)
# print(p.mean)
