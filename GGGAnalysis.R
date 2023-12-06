library(naivebayes)
library(discrim)
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(bonsai)

## Read in the data
GGGTrain <- vroom("train.csv")
GGGTest <- vroom("test.csv")

## Recipe
ggg_recipe <- recipe(type~., data=GGGTrain) %>%
  step_lencode_glm(color, outcome=vars(type))

## Apply the recipe to your data
prep <- prep(ggg_recipe)
baked <- bake(prep, new_data = GGGTrain)

## Random Forest
my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(my_mod)

# Set up grid of tuning values 
tuning_grid <- grid_regular(mtry(range =c(1,7)), min_n(), levels = 3) 

# Set up K-fold CV
folds <- vfold_cv(GGGTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- forest_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
final_forest_wf <-
  forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGTrain)

## Predict
final_forest_wf %>%
  predict(new_data = GGGTrain, type="class")

forest_predictions <- predict(final_forest_wf,
                              new_data=GGGTest,
                              type="class") # "class" or "prob" (see doc)

vroom_write(x=forest_predictions, file="forest.csv", delim=",")

## Naive Bayes
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
# Set up grid of tuning values 
tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 3)

# Find best tuning parameters 
CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
final_bayes_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGTrain)

## Predict
final_bayes_wf %>%
  predict(new_data = GGGTrain, type="class")

bayes_predictions <- predict(final_bayes_wf,
                             new_data=GGGTest,
                             type="class") # "class" or "prob" (see doc)

vroom_write(x=bayes_predictions, file="bayes.csv", delim=",")

## Neural Networks
nn_recipe <- recipe(type~., data=GGGTrain) %>%
  update_role(id, new_role="id") %>%
  step_lencode_glm(color, outcome=vars(type)) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
  epochs = 50) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=10)

folds <- vfold_cv(GGGTrain, v = 5, repeats = 1)

nn_wf <-workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

tuned_nn <- nn_wf %>%
  tune_grid(
    resamples = folds,
    grid = nn_tuneGrid,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE)
  )

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## Find Best Tuning Parameters
bestTune <- tuned_nn %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
final_nn_wf <-
  nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGTrain)

## Predict
final_nn_wf %>%
  predict(new_data = GGGTrain, type="class")

nn_predictions <- predict(final_nn_wf,
                             new_data=GGGTest,
                             type="class") # "class" or "prob" (see doc)

vroom_write(x=nn_predictions, file="nn.csv", delim=",")

## Boosting and Bart
boost_recipe <- recipe(type~., data=GGGTrain) %>%
  update_role(id, new_role="id") %>%
  step_lencode_glm(color, outcome=vars(type)) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

# Model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

boost_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=10)

folds <- vfold_cv(GGGTrain, v = 5, repeats = 1)

boost_wf <-workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)

tuned_boost <- boost_wf %>%
  tune_grid(
    resamples = folds,
    grid = boost_tuneGrid,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE)
  )

## CV tune, finalize and predict here and save results
## Find Best Tuning Parameters
bestTune <- tuned_bart %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
final_bart_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGGTrain)

## Predict
final_bart_wf %>%
  predict(new_data = GGGTrain, type="class")

bart_predictions <- predict(final_bart_wf,
                          new_data=GGGTest,
                          type="class") # "class" or "prob" (see doc)

vroom_write(x=bart_predictions, file="bart.csv", delim=",")