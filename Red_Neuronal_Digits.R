###########################################################
#######**************** LIBRERIAS ****************#########
###########################################################
install.packages("e1071")
install.packages("caret")
install.packages("jpeg")
install.packages("psych")
install.packages("tidyr")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("neuralnet")
install.packages("nnet")
install.packages("RCurl")
install.packages("Metrics")

library(e1071)
library(caret)
library(jpeg)
library("psych")
library(tidyr)
library(readr)
library(dplyr)
library(corrplot)
library("ggplot2")
library(neuralnet)
library(nnet)
library(RCurl)
library(Metrics)

##############################################################################
#############**CARGA DE INFORMACION -- http://yann.lecun.com/********#########
##############################################################################

# modification of https://gist.github.com/brendano/39760
# automatically obtains data from the web
# creates two data frames, test and train
# labels are stored in the y variables of each data frame
# can easily train many models using formula `y ~ .` syntax

# download data from http://yann.lecun.com/exdb/mnist/
download.file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
              "train-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
              "train-labels-idx1-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
              "t10k-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
              "t10k-labels-idx1-ubyte.gz")

######################################################
####*DESCEOMPRESION DE INFORMACION****######
######################################################


# gunzip the files
R.utils::gunzip("train-images-idx3-ubyte.gz")
R.utils::gunzip("train-labels-idx1-ubyte.gz")
R.utils::gunzip("t10k-images-idx3-ubyte.gz")
R.utils::gunzip("t10k-labels-idx1-ubyte.gz")


##################################################################################################
####*DEFINICION DE FUNCIONES DE VISUALIZACION Y TRANSFORMACION DE INFORMACION*****######
##################################################################################################

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

?readBin

# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x     = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}


###########################################################
####*CARGA DE INFORMACION EN DATAFRAME*****######
###########################################################

# Carga las imagenes
# en las variables train y test
train = load_image_file("train-images-idx3-ubyte")
test  = load_image_file("t10k-images-idx3-ubyte")

# Carga las etiquetas
# adicionandolas en las train y test
train$y = as.factor(load_label_file("train-labels-idx1-ubyte"))
test$y  = as.factor(load_label_file("t10k-labels-idx1-ubyte"))


## unificacion de la data en un solo dataset de 60.000 train+10.000 test
Total = rbind(train,test)
dim(Total)

##########################################
#######**** *DESCRIPTIVA* ****############
##########################################

## Descriptiva

ggplot(Total, aes(x=y,fill=as.factor(y))) +
  geom_histogram(stat = "count")

summary(Total$y)


#####################################################################
########*SELECCION DATOS DE ENTRENAMIENTO Y VALIDACIÓN *****#########
#####################################################################

##se divide el archivo de 70.000 registros en entrenamiento y validación 
#se agrega una semilla para que el muestreo sea replicable

set.seed(15723)
sample<-sample.int(nrow(Total), floor(0.99*nrow(Total)))
Dat.train<-Total[sample,]
Dat.test<-Total[-sample,]

sample2<-sample.int(nrow(Dat.test), floor(0.6*nrow(Dat.test)))
Entrenamiento <- Dat.test[sample2,]
validacion <- Dat.test[-sample2,]

#Verificamos el balance de la variable objetivo en los conjuntos de entrenamiento y validación
table(Entrenamiento$y)
table(validacion$y)

###########################################################
##################***** PREPARACIÓN ******#################
###########################################################

#La codificación de las variables categóricas es necesaria cuando se usa neuralnet, ya que no trabaja con factores

str(Entrenamiento$y)
str(validacion$y)
head(Entrenamiento$y)

#La codificación de las variables categóricas es necesaria cuando se usa neuralnet, ya que no trabaja con factores
# Encode para representar la variable objetivo como númerica
NNTrain <- cbind(Entrenamiento[, 1:784], class.ind(Entrenamiento$y))

# Configuración las etiquetas de los distintos valores que toma la variable objetivo
names(NNTrain) <- c(names(Entrenamiento)[1:784],"y0","y1","y2","y3","y4","y5","y6","y7","y8","y9")

# Aunque la Normalización no es necesaria se hizo para estandarizar un poco los datos
#NN_Train <- NNTrain[,1:784]/255
#NN_Val <- validacion[,1:784]/255

# Set up formula
n <- names(NNTrain)
f <- as.formula(paste("y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 ~", paste(n[!n %in% c("y0","y1","y2","y3","y4","y5","y6","y7","y8","y9")], collapse = " + ")))
f

###########################################################
#######**** IMPLEMENTACIÓN DE LA RED NEURONAL *****########
###########################################################

nn <- neuralnet(f,
                data = NNTrain,
                hidden = c(400),
                act.fct = "logistic",
                linear.output = FALSE)

## hidden: 13, 10, 3    thresh: 0.01    rep: 1/1    steps:      88  error: 0.03039  time: 0.1 secs

plot(nn)

# Compute predictions
pr.nn <- compute(nn, NNTrain[, 1:784])

# Extract results
pr.nn_ <- pr.nn$net.result
head(pr.nn_)


original_values <- max.col(NNTrain[, 785:794])
pr.nn_2 <- max.col(pr.nn_)
mean(pr.nn_2 == original_values)


###########################################################
#######**** VALIDACIÓN CRUZADA EN RED NEURONAL *****#######
###########################################################



# Implementaremos validación cruzada para evitar el sobreajuste

set.seed(32323)
# 10 fold cross validation
k <- 5
# Results from cv
outs <- NULL
# Train test split proportions
proportion <- 0.95

# Crossvalidate, go!
for(i in 1:k)
{
  index <- sample(1:nrow(NNTrain), round(proportion*nrow(NNTrain)))
  train_cv <- NNTrain[index, ]
  test_cv <- NNTrain[-index, ]
  nn_cv <- neuralnet(f,
                     data = train_cv,
                     hidden = c(200),
                     act.fct = "logistic",
                     linear.output = FALSE)
  
  # Compute predictions
  pr.nn <- compute(nn_cv, validacion[, 1:784])
  # Extract results
  pr.nn_ <- pr.nn$net.result
  # Accuracy (test set)
  original_values <- max.col(validacion[, 785:794])
  pr.nn_2 <- max.col(pr.nn_)
  outs[i] <- mean(pr.nn_2 == original_values)
}

mean(outs)
## [1] 0.9888888889
