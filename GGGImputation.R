library(tidyverse)
library(tidymodels)
library(vroom)

## Read in the data
GGGMissingTrain <- vroom("trainWithMissingValues.csv")
GGGTrain <- vroom("train.csv")
GGGTest <- vroom("test.csv")

## Recipe
gggmiss_recipe <- recipe(type~., data=GGGMissingTrain) %>%
  step_impute_median(all_numeric_predictors())

## Apply the recipe to your data
prep <- prep(gggmiss_recipe)
baked <- bake(prep, new_data = GGGMissingTrain)

## RMSE
rmse_vec(GGGTrain[is.na(GGGMissingTrain)], baked[is.na(GGGMissingTrain)])
