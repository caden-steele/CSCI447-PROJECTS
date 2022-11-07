
# Read in data from an Excel .csv file
# read.csv("cancer.csv")
labCSV <- read.csv("cancer.csv")
labCSV  
names(labCSV)

# Fit four SLR models:  X is population, Y is mortality 
# Model 1 Origingal X and Y data (no transformations)
# Model 2 Log transform X only
# Model 3 Log transform Y only
# Model 4 Log transform both X and Y

population <- read.csv("cancer.csv")
mortality <- read.csv("cancer.csv")

# Take log transformations of both variables
log.population = log(population)
log.mortality = log(mortality)


#############################################################

lm.out <- lm(population ~ mortality)           # Model 1
#lm.out <- lm(mortality ~ log.population)       # Model 2
#lm.out <- lm(log.mortality ~ population)       # Model 3
#lm.out <- lm(log.mortality ~ log.population)   # Model 4
summary(lm.out)

# Plot the data with the regression line
windows()  
par(mfrow=c(1,3))
plot(population,mortality,main="Model 1: Original Data",pch=16,cex=1.5)   # Model 1
#plot(log.population,mortality,main="Model 2: ln(X) Data",pch=16,cex=1.5)  # Model 2
#plot(population,log.mortality,main="Model 3: ln(Y) Data",pch=16,cex=1.5)  # Model 3
#plot(log.population,log.mortality,main="Model 4: ln(X) and ln(Y) Data",pch=16,cex=1.5)   # Model 4
abline(lm.out)

# Make residual plots
plot(fitted(lm.out),resid(lm.out),pch=16,cex=1.5,main="Residuals vs Predicted Values")
abline(h=0)
qqnorm(resid(lm.out),pch=16,cex=1.5,main="Normal Probability Plot")
abline(h=0)
