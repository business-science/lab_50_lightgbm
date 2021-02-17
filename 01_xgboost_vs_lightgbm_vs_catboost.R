# BUSINESS SCIENCE LEARNING LABS ----
# LAB 50: LIGHTGBM VS XGBOOST VS CATBOOST ----
# MODULE 01: MODEL COMPARISONS ---- 
# **** ----

# LEARNING GOALS ----
# - Exposure to Treesnip (Tidymodels Interface to LightGBM & CatBoost)

# **** ----

# GITHUB INSTALLATIONS ----
# - TREESNIP: remotes::install_github("curso-r/treesnip")
# - CATBOOST: devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')

# LIBRARIES ----

# Machine Learning Packages
library(lightgbm)
library(catboost)
library(xgboost)

# Tidymodels
library(treesnip)
library(tidymodels)

# Core
library(tidyverse)


# LIGHTGBM BASIC USAGE ----

data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')

train  <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)

model <- lgb.train(
    params = list(
        objective = "regression" , metric = "l2"
    ), 
    data = dtrain
)

predict(model, agaricus.test$data)



# 1.0 AGARICUS ----

# * Data Prep ----
agaricus_train_tbl <- agaricus.train$data %>% 
    as.matrix() %>% 
    as_tibble() %>%
    add_column(target = agaricus.train$label, .before = 1) %>%
    mutate(target = factor(target))

agaricus_test_tbl <- agaricus.test$data %>% 
    as.matrix() %>% 
    as_tibble() %>%
    add_column(target = agaricus.test$label, .before = 1) %>%
    mutate(target = factor(target))


# * Helper Functions ----
train_model <- function(model_spec, train = agaricus_train_tbl) {
    workflow() %>%
        add_model(spec = model_spec) %>%
        add_recipe(recipe = recipe(target ~ ., train)) %>%
        fit(train)
}

make_predictions <- function(model, test = agaricus_test_tbl, type = "prob") {
    predict(model, test, type = type) %>%
        bind_cols(
            test %>% select(target)
        ) %>%
        mutate(target = factor(target))
}

extract_fitted_model <- function(model) {
    model %>% 
        pull_workflow_fit() %>%
        pluck("fit")
}



# * LightGBM ----

agaricus_lightgbm_fit_wflw <- boost_tree(
    mode       = "classification", 
    learn_rate = 2
) %>%
    set_engine("lightgbm") %>%
    train_model(agaricus_train_tbl)

agaricus_lightgbm_fit_wflw %>%
    make_predictions(agaricus_test_tbl, type = "prob") %>%
    yardstick::roc_auc(target, .pred_1, event_level = "second")

agaricus_lightgbm_fit_wflw %>% 
    extract_fitted_model() %>%
    lightgbm::lgb.importance() %>%
    lightgbm::lgb.plot.importance()


# * XGBoost ----

agaricus_xgboost_fit_wflw <- boost_tree(mode = "classification") %>%
    set_engine("xgboost") %>%
    train_model(agaricus_train_tbl)

agaricus_xgboost_fit_wflw %>%
    make_predictions(agaricus_test_tbl, type = "prob") %>%
    yardstick::roc_auc(target, .pred_1, event_level = "second")

agaricus_xgboost_fit_wflw %>% 
    extract_fitted_model() %>%
    xgboost::xgb.importance(model = .) %>%
    xgboost::xgb.plot.importance()

# * CatBoost ----

agaricus_catboost_fit_wflw <- boost_tree(mode = "classification") %>%
    set_engine("catboost") %>%
    train_model(agaricus_train_tbl)

agaricus_catboost_fit_wflw %>%
    make_predictions(agaricus_test_tbl, type = "prob") %>%
    yardstick::roc_auc(target, .pred_1, event_level = "second")

agaricus_catboost_fit_wflw %>% 
    extract_fitted_model() %>%
    catboost::catboost.get_feature_importance() %>%
    as_tibble(rownames = "feature") %>%
    rename(value = V1) %>%
    arrange(-value) %>%
    mutate(feature = as_factor(feature) %>% fct_rev()) %>%
    dplyr::slice(1:10) %>%
    ggplot(aes(value, feature)) +
    geom_col()
    


# 2.0 DIAMONDS ----

diamonds

set.seed(123)
diamonds_splits <- vfold_cv(diamonds, v = 5)

recipe_spec     <- recipe(price ~ ., data = diamonds)

# * XGBoost ----
doParallel::registerDoParallel(8)
resamples_xgboost_tbl <- workflow() %>%
    add_model(boost_tree(mode = 'regression') %>% set_engine("xgboost")) %>%
    
    # IMPORTANT NOTE - XGBoost cannot handle factor (or ordered) data. Must convert to Dummy.
    add_recipe(recipe_spec %>% step_dummy(all_nominal())) %>%
    
    fit_resamples(
        resamples = diamonds_splits,
        control   = control_resamples(verbose = TRUE, allow_par = FALSE)
    )

# * LightGBM ----
doParallel::registerDoParallel(8)
resamples_lightgbm_tbl <- workflow() %>%
    add_model(boost_tree(mode = 'regression') %>% set_engine("lightgbm")) %>%
    add_recipe(recipe_spec) %>%
    fit_resamples(
        resamples = diamonds_splits,
        control   = control_resamples(verbose = TRUE, allow_par = TRUE)
    )

# * Catboost ----
doParallel::registerDoParallel(8)
resamples_catboost_tbl <- workflow() %>%
    add_model(boost_tree(mode = 'regression') %>% set_engine("catboost")) %>%
    add_recipe(recipe_spec) %>%
    fit_resamples(
        resamples = diamonds_splits,
        control   = control_resamples(verbose = TRUE, allow_par = TRUE)
    )



# * Comparison ----

bind_rows(
    resamples_catboost_tbl %>% collect_metrics() %>% add_column(.model = "catboost"),
    resamples_lightgbm_tbl %>% collect_metrics() %>% add_column(.model = "lightgbm"),
    resamples_xgboost_tbl %>% collect_metrics() %>% add_column(.model = "xgboost")
)


# CONCLUSIONS ----
# - The algorithms vary significantly out of the box
# - Experimentation with multiple algorithms is very important

