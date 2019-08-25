# data cleaning - create the same clean datasets for R Caret and Python Sklearn comparison

library(tidyverse)
library(stringr)
library(magrittr)

dat_raw <- read_csv("./data/clinvar_conflicting.csv")

glimpse(dat_raw)

# check missing value, unique values, etc.

fct_dat_summary <- . %>%
    summarise_all(list(
        fct_n_missing = ~sum(is.na(.)),
        fct_n_unq = ~unique(.) %>% length,
        fct_unq_val10 = ~unique(.) %>% na.omit %>% head(n = 10) %>% str_c(collapse = ", "),
        fct_dtype = ~class(.))) %>%
    gather(key = var, value = value) %>%
    separate(col = var, into = c("var", "fct"), sep = "_fct_") %>%
    spread(key = fct, value = value)


dat_summary <- fct_dat_summary(dat_raw)

dat_summary %<>% 
    mutate(n_missing = n_missing %>% as.numeric,
           n_unq = n_unq %>% as.numeric)

# drop variables with too many missing values (e.g. > 5000)
# drop character variables with too many unique categories (e.g. > 100)
var_keep <- dat_summary %>% 
    filter(!(n_missing > 5000 | (dtype == "character" & n_unq > 100))) %>% 
    pull(var)

# there are 58543 data left, good size for experiment
dat_clean <- dat_raw %>%
    select(var_keep) %>%
    na.omit()

dat_clean %>% fct_dat_summary() %>% arrange(n_unq)

# drop biotype and feature_type - only take 1 value for all records
dat_clean %<>% set_names(tolower(names(.))) %>%
    select(-biotype, -feature_type, -consequence) %>%
    mutate(class = as.factor(class))



# create training and testing data ----------------------------------------

set.seed(1234)
idx_train <- tibble(idx = 1:nrow(dat_clean)) %>% 
    sample_frac(.7, replace = F) %>%
    pull(idx)

# 70% random sample for training
dat_train <- dat_clean[idx_train, ]
# 30% random sample for testing
dat_test <- dat_clean[-idx_train, ] 


# save to disk ------------------------------------------------------------

dat_train %>% write_csv("./data/clinvar_conflicting_train.csv")
dat_test %>% write_csv("./data/clinvar_conflicting_test.csv")