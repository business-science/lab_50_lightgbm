# BUSINESS SCIENCE LEARNING LABS ----
# LAB 50: LIGHTGBM ----
# MODULE 02: HIERARCHICAL FORECASTING WITH LIGHTGBM & FRIENDS ---- 
# **** ----

# BUSINESS OBJECTIVE ----
# - Forecast intermittent demand 
# - Predict next 28-Days

# **** ----

# GITHUB INSTALLATIONS ----
# - TREESNIP: remotes::install_github("curso-r/treesnip")
# - CATBOOST: devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')

# LIBRARIES ----

# Machine Learning
library(lightgbm)
library(catboost)
library(xgboost)

# Tidymodels
library(treesnip)
library(modeltime)
library(modeltime.ensemble)
library(tidymodels)

# Core
library(skimr)
library(timetk)
library(tidyverse)


# DATA -----

# * Read Data ----
calendar_tbl <- read_csv("m5-forecasting-accuracy/calendar.csv")
calendar_tbl

sales_sample_tbl <- read_rds("m5-forecasting-accuracy/sales_sample_tbl.rds")
sales_sample_tbl

hierarchy_tbl <- sales_sample_tbl %>% select(contains("id"))
hierarchy_tbl
    

# * Reshape & Join ----

sales_sample_long_tbl <- sales_sample_tbl %>%
    pivot_longer(
        cols      = starts_with("d_"), 
        names_to  = "day", 
        values_to = "value"
    ) %>%
    left_join(calendar_tbl, by = c("day" = "d")) %>%
    select(contains("_id"), date, value)

# * Skim Data ----

skimr::skim(sales_sample_long_tbl)

# * Visualize ----

set.seed(123)
item_id_sample <- sample(unique(sales_sample_long_tbl$item_id), size = 6)

sales_sample_long_tbl %>%
    filter(item_id %in% item_id_sample) %>%
    group_by(item_id) %>%
    plot_time_series(
        date, value, 
        .smooth        = TRUE, 
        .smooth_period = 28, 
        .facet_ncol    = 2
    )

# PREPARE FULL DATA ----

full_data_tbl <- sales_sample_long_tbl %>%
    
    # PERFORM HIERARCHICAL AGGREGATIONS
    add_column(all_stores_id = "all_stores", .before = 1) %>%
    pivot_longer(
        cols      = ends_with("_id"), 
        names_to  = "category", 
        values_to = "identifier" 
    ) %>%
    
    group_by(category, identifier, date) %>%
    summarise(value = sum(value, na.rm = TRUE)) %>%
    ungroup() %>%
    
    # APPLY TIME SERIES FEATURE ENGINEERING 
    group_by(category, identifier) %>%
    
    # Fix any missing timestamps
    pad_by_time(date, .by = "day", .pad_value = 0) %>%
    
    # Extend into future 28-days per item id
    future_frame(date, .length_out = 28, .bind_data = TRUE) %>%
    
    # Add time series features
    tk_augment_lags(value, .lags = 28) %>%
    tk_augment_slidify(
        value_lag28,
        .f       = ~ mean(., na.rm = TRUE),
        .period  = c(7, 14, 28, 28*2),
        .align   = "center",
        .partial = TRUE
    ) %>%
    
    ungroup() %>%
    
    rowid_to_column(var = "row_id") 

full_data_tbl %>% glimpse()

full_data_tbl %>% skim()

# full_data_tbl %>% write_rds("m5-forecasting-accuracy/full_data_tbl.rds")

full_data_tbl <- read_rds("m5-forecasting-accuracy/full_data_tbl.rds")

# OPTIONAL - ADD HIERARCHICAL FEATURES ----
# - Tried with and without - Didn't improve very much
#   - Why? Tree-based models are naturally hierarchical
#   - Similar products naturally get lumped together
#   - Similar locations naturally get lumped together
# - Other alternatives (not explored)
#   - Create 2 model strategies: aggregations and product-level

# full_data_tbl <- full_data_tbl %>%
#     left_join(
#         hierarchy_tbl %>% select(-id),
#         by = c("identifier" = "item_id")
#     )

# DATA PREPARED & FUTURE DATA ----

data_prepared_tbl <- full_data_tbl %>% 
    filter(!is.na(value)) %>%
    filter(!is.na(value_lag28))

data_prepared_tbl %>% skim()

future_data_tbl <- full_data_tbl %>%
    filter(is.na(value))

future_data_tbl %>% skim()

# TIME SPLIT ----

splits <- data_prepared_tbl %>%
    time_series_split(date, assess = 28, cumulative = TRUE)

splits %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, value)

# RECIPE ----

recipe_spec <- recipe(value ~ ., data = training(splits)) %>%
    update_role(row_id, date, new_role = "id") %>%
    step_timeseries_signature(date) %>%
    step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
    step_dummy(all_nominal(), one_hot = TRUE)

recipe_spec %>% summary()

recipe_spec %>% prep() %>% juice() %>% glimpse()


# MACHINE LEARNING ----

# * LIGHTGBM ----

wflw_lightgbm_defaults <- workflow() %>%
    add_model(
        boost_tree(mode = "regression") %>%
            set_engine("lightgbm")
    ) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_lightgbm_tweedie <- workflow() %>%
    add_model(
        boost_tree(mode = "regression") %>%
            set_engine("lightgbm", objective = "tweedie")
    ) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

# * XGBOOST ----

wflw_xgboost_defaults <- workflow() %>%
    add_model(
        boost_tree(mode = "regression") %>%
            set_engine("xgboost")
    ) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_xgboost_tweedie <- workflow() %>%
    add_model(
        boost_tree(mode = "regression") %>%
            set_engine("xgboost", objective = "reg:tweedie")
    ) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

# * CATBOOST ----

wflw_catboost_defaults <- workflow() %>%
    add_model(
        boost_tree(mode = "regression") %>%
            set_engine("catboost")
    ) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_catboost_tweedie <- workflow() %>%
    add_model(
        boost_tree(mode = "regression") %>%
            set_engine("catboost", loss_function = "Tweedie:variance_power=1.5")
    ) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))


# MODELTIME ----

# * Calibrate on Test ----
calibration_tbl <- modeltime_table(
    wflw_lightgbm_defaults,
    wflw_xgboost_defaults,
    wflw_catboost_defaults,
    wflw_lightgbm_tweedie,
    wflw_xgboost_tweedie,
    wflw_catboost_tweedie
) %>%
    modeltime_calibrate(testing(splits)) %>%
    mutate(.model_desc = ifelse(.model_id > 3, str_c(.model_desc, " - Tweedie"), .model_desc))

calibration_tbl %>% modeltime_accuracy()

# * Forecast Test ----

test_forecast_tbl <- calibration_tbl %>%
    modeltime_forecast(
        new_data    = testing(splits),
        actual_data = data_prepared_tbl,
        keep_data   = TRUE 
    )

# * Create Visualization Filters -----
#   - Switch these out

# - All Stores Aggregated
filter_identfiers_all <- "all_stores"

# - State-Level Forecasts
filter_identfiers_state <- full_data_tbl %>%
    filter(category == "state_id") %>%
    distinct(identifier) %>%
    pull()

# - Item-Level Forecasts (Sample of 6)
filter_identfiers_items <- item_id_sample 

# * Visualize ----

test_forecast_tbl %>%
    
    # FILTER IDENTIFIERS
    filter(identifier %in% filter_identfiers_all) %>%
    
    group_by(identifier) %>%
    
    # Focus on end of series
    filter_by_time(
        .start_date = last(date) %-time% "3 month", 
        .end_date = "end"
    ) %>%
    
    plot_modeltime_forecast(
        .facet_ncol         = 2, 
        .conf_interval_show = FALSE,
        .interactive        = TRUE
    )

# * Accuracy by Identifier ----
accuracy_by_identifier_tbl <- test_forecast_tbl %>%
    select(category, identifier, .model_desc, .index, .value) %>%
    pivot_wider(
        names_from   = .model_desc,
        values_from  = .value
    ) %>%
    filter(!is.na(LIGHTGBM)) %>%
    pivot_longer(cols = LIGHTGBM:`CATBOOST - Tweedie`) %>%
    group_by(category, identifier, name) %>%
    summarize_accuracy_metrics(
        truth      = ACTUAL, 
        estimate   = value, 
        metric_set = default_forecast_accuracy_metric_set()
    )

best_rmse_by_indentifier_tbl <- accuracy_by_identifier_tbl %>%
    group_by(identifier) %>%
    slice_min(rmse, n = 1) %>%
    ungroup()

best_rmse_by_indentifier_tbl %>% View()

best_rmse_by_indentifier_tbl %>% 
    group_by(category) %>%
    summarise(median_rmse = median(rmse))

best_rmse_by_indentifier_tbl %>% count(name, sort = TRUE)

# ENSEMBLE ----

# * Make an Ensemble ----

ensemble_tbl <- calibration_tbl %>%
    filter(.model_id %in% c(2, 5)) %>%
    ensemble_weighted(loadings = c(2, 3)) %>%
    modeltime_table()

test_forecast_ensemble_tbl <- ensemble_tbl %>%
    modeltime_calibrate(testing(splits)) %>%
    modeltime_forecast(
        new_data    = testing(splits),
        actual_data = data_prepared_tbl,
        keep_data   = TRUE 
    )

# test_forecast_ensemble_tbl %>% 
#     write_rds("m5-forecasting-accuracy/test_forecast_ensemble_tbl.rds")

# * Refit Ensemble ----

ensemble_refit_tbl <- ensemble_tbl %>%
    modeltime_refit(data_prepared_tbl)

future_forecast_ensemble_tbl <- ensemble_refit_tbl %>%
    modeltime_forecast(
        new_data    = future_data_tbl,
        actual_data = data_prepared_tbl,
        keep_data   = TRUE 
    )

# future_forecast_ensemble_tbl %>% 
#     write_rds("m5-forecasting-accuracy/future_forecast_ensemble_tbl.rds")

# * Visualize Future Forecast ----

future_forecast_ensemble_tbl %>%
    
    # FILTERS 
    filter(identifier %in% filter_identfiers_items) %>%
    
    group_by(identifier) %>%
    
    # Focus on end of series
    filter_by_time(
        .start_date = last(date) %-time% "6 month", 
        .end_date = "end"
    ) %>%
    
    plot_modeltime_forecast(
        .facet_ncol         = 2, 
        .conf_interval_show = TRUE,
        .interactive        = TRUE
    )

# CONCLUSIONS ----
# - Hierarchical Forecasting is a challenging problem
# - XGBoost was the winner for this data set, but experimentation is critical
# - Didn't cover:
#   - Hyperparameter Tuning (Probably why CatBoost & LightGBM aren't as good)
#   - More sophisticated ensembles
#   - Deep Learning (GluonTS)
# - I cover these topics at length in my:
#       High-Performance Time Series Forecasting Course (DS4B 203-R)

