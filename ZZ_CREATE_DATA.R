# SCRIPTS TO CREATE A SAMPLE OF THE M5 COMPETITION DATA

library(vroom)
library(tidyverse)

sales_train_tbl <- vroom::vroom("m5-forecasting-accuracy/sales_train_evaluation.csv")

set.seed(123)
sales_sample_tbl <- sales_train_tbl %>%
    sample_n(size = 50) 

sales_sample_tbl %>% write_rds("m5-forecasting-accuracy/sales_sample_tbl.rds")