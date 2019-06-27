# importing the dataset
dataset = read.csv('Data.csv')

#missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
#recorre la columna age y se fija si una casilla esta vacia. Si esta vacia calcula
#el promedio de toda la columna, si no deja el valor como estaba
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)