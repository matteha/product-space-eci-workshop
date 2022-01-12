see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest
see .org file for latest

# Jupyter: no conflicts with packages (M1 OS X, pre-compiled)
# Need e.g. quartz for imager using brew
# Many dependencies, co lab solves this
# Install packages (on google colab)
# need to set this in google colab? to not get interactive prompt
# options(install.packages.compile.from.source = "always")
# set to noone when on M1 Mac
# options(install.packages.compile.from.source = "always")
print(installed.packages())
install.packages('highcharter')
install.packages('maps')
install.packages('tidyverse')
install.packages('reticulate') 
install.packages('foreign')
install.packages('igraph')
install.packages('imager')
install.packages('economiccomplexity')
install.packages('tidylog') # for join (merge) statistics, e.g. left_only
install.packages('arrow')

# Load libraries necessary for workshop
library('foreign') # to load STATA files
library('glue') # to concatenate strings / variables
library('igraph') # network analysis
library('imager') # to load / show images
library(tidylog, warn.conflicts = FALSE) # for join statistics, e.g. left_only
library('economiccomplexity')
library("reticulate") # R - Python interaction
# -- if reticulate throws error on OSX:
# -- brew install xquartz --cask
library('highcharter')
library('maps')
library('arrow')
library('tidyverse') # data analysis standard toolkit
# -- readr: for reading data
# -- ggplot2: for plotting
# -- tibble: for creating “tibbles”; these are the tidyverse’s take on data frames.
# -- dplyr: for manipulating tibbles (or data frames); creating new variables, calculating summary statistics etc.
# -- tidyr: for reshaping data (making it from long to wide format, and vice versa)
# -- purrr: for functional programming.
# -- stringr: for manipulating strings
# -- forcats: FOR CATegorical data (factors); this makes it easier to reorder and rename the levels in factor variables.
# -- library(datasets) # - Built in datasets
print('loaded libraries')

# load trade data, parquet file as tibble
# -- this is a large file because it includes strings, takes 1 - 3 minutes to load
# -- can also merge strings in separately but easier for illustrative purposes to have them all
# -- preloaded (R quickly runs into memory problems when merging)
# -- without labels: df_orig <- read_csv("https://www.dropbox.com/s/42xfb2oq8ju54b4/trade_without_labels.csv?dl=1")
# -- df_orig <- read_parquet("/Users/admin/Dropbox/proj/org_zhtml_projects/product-space-eci-workshop/files/trade_without_labels.parquet", as_tibble = TRUE)
# -- df_orig <- read_parquet("https://www.dropbox.com/s/xf4azh8eelk2etd/trade_without_labels.parquet?dl=1", as_tibble=TRUE)
df_orig <- read_csv('https://www.dropbox.com/s/3n4r4qo4j0jjpln/trade.csv?dl=1')

# Explore data
head(df_orig)
sample_n(df_orig,10)
dim(df_orig)

unique(df_orig$year)

length(unique(df_orig$product_name))

# str_detect(df_orig$country_name,'Netherland')
# dplyr::filter(df_orig, !grepl("Netherland",country_name))
# filter(df_orig, !grepl("Netherland",country_name))$country_name

# Chaining commands
# %>% command: you can think of this as saying 'then' 
# distinct is built within dplyr (rather than unique())

df_orig %>% 
  filter(grepl('Netherland', country_name)) %>% 
  distinct(country_name)

# ignore case (see documentation, K1 in emacs / vim)
df_orig %>% 
  filter(grepl('wine', product_name, ignore.case = TRUE)) %>% 
  distinct(product_name)

# df2 <- df_orig[ (df_orig$country_code=='USA') & (df_orig$year==2012) ]
df_orig %>%
  filter(country_code == 'USA' & year == 2012) %>%
  group_by(product_code,product_name) %>%
  summarise(sum_export_value = sum(export_value)) %>%
  arrange(desc(sum_export_value)) %>%
  head()

# Plot
df_orig %>%
  filter(country_code == 'USA' & product_code == 8703) %>%
  ggplot(aes(x = year, y = export_value)) + geom_line()

df2 <- filter(df_orig,country_code == 'USA' & product_code == 8703)
df2 %>% ggplot(aes(x = year, y = export_value)) + geom_line()

# RCA

ORIGINAL WITH symbols

  #calc_rca <- function(data,country_col,product_col,time_col,value_col) {
  calc_rca <- function(data,region_col,product_col,time_col,value_col) {
      # Calculates: Revealed Comparative Advantage (RCA) of country-product-time combinations
      # Returns: pandas dataframe with RCAs
      # turn strings into symbols to use in tidyr
      # https://github.com/r-lib/rlang/issues/116
      # https://stackoverflow.com/questions/48062213/dplyr-using-column-names-as-function-arguments
      # !! doesn't do anything by itself and is not a real operator, it tells mutate() to do something though, because mutate() is designed to recognize it.
      # What it tells to mutate() is to act as if !!x was replaced by the quoted content of x.
      # https://stackoverflow.com/questions/57136322/what-does-the-operator-mean-in-r-particularly-in-the-context-symx
      # country_col2 <- sym(country_col)
      # See the !!; they mean “hey R, remember the expression I stored recently? Now take it, and ‘unquote’ it, that is, just run it!”. The double exclamation mark is just syntactic sugar for that phrase.
      # create all country-product-time combinations
      # https://stackoverflow.com/questions/29678435/how-to-pass-dynamic-column-names-in-dplyr-into-custom-function
      # You can use .data inside dplyr chain now.
      # library(dplyr)
      # from <- "Stand1971"
      # to <- "Stand1987"
      # data %>% mutate(diff = .data[[from]] - .data[[to]])
      # Another option is to use sym with bang-bang (!!)
      # data %>% mutate(diff = !!sym(from) - !!sym(to))
      df_all <- data %>% 
        tidyr::expand(!!sym(country_col), !!sym(product_col), !!sym(time_col)) 
      # rename columns accordingly
      names(df_all) <- c(region_col,product_col,time_col) 
      # merge original data back into all-combinations
      df_all <- left_join(df_all,data,by=c('time_col','country_col','product_col'))
      # set export value to 0 if missing (fills in the extra combinations created)
      df_all %>% 
        dplyr::mutate(value_col = replace_na(!!sym(value_col), 0))
      # set to numeric again
      # df_all$!!sym(value_col) = as.factor(df_all$!!sym(value_col))
      #df_all %>% 
        #mutate(across(!!sym(value_col), factor))
      # define RCA properties
      df_all <- df_all %>% mutate(Xcpt = !!sym(value_col))
      #####
      return(df_all)
  }
  define 'region_col'
  run function
  df_rca = calc_rca(data=df_orig,
                    country_col='country_name',
                    product_col='product_name',
                    time_col='year',
                    value_col='export_value')

calc_rca <- function(data,region_col,product_col,time_col,value_col) {
    # - add all possible products for each country with export value 0
    # - else matrices later on will have missing values in them, complicating calculations
    df_all <- data %>% 
    expand(time_col,region_col, product_col) 
    # merge data back in
    df_all <- left_join(df_all,data,by=c('time_col','region_col','product_col'))
    # set export value to 0 if missing (fills in the extra combinations created)
    df_all <- df_all %>% 
      mutate(value_col = replace_na(value_col, 0))
    # define RCA properties
    # -- Xcpt
    df_all <- df_all %>% mutate(Xcpt = value_col)
    # -- Xct
    df_all <- df_all %>% 
      group_by(region_col,time_col) %>% 
      mutate(Xct = sum(value_col)) 
    # -- Xct
    df_all <- df_all %>% 
      group_by(time_col,region_col) %>% 
      mutate(Xct = sum(value_col)) 
    # -- Xpt
    df_all <- df_all %>% 
      group_by(time_col,product_col) %>% 
      mutate(Xpt = sum(value_col)) 
    # -- Xt
    df_all <- df_all %>% 
      group_by(time_col) %>% 
      mutate(Xt = sum(value_col)) 
    # -- RCAcpt
    df_all$RCAcpt = (df_all$Xcpt/df_all$Xct)/(df_all$Xpt/df_all$Xt)
    # set RCAcpt to 0 if missing, e.g. if product / country have 0 (total) exports
    df_all <- df_all %>% 
      dplyr::mutate(RCAcpt = replace_na(RCAcpt, 0))
    # drop the properties 
    #df_all.drop(['Xcpt','Xct','Xpt','Xt'],axis=1,inplace=True,errors='ignore')
    df_all <- select(df_all, -c(Xcpt,Xct,Xpt,Xt))
    #####
    return(df_all)
}
# rename columns accordingly
df_rca <- df_orig %>%
  rename(time_col = year,
         region_col = country_name,
         product_col = product_name,
         value_col = export_value)
# calculate RCA
df_rca <- calc_rca(data=df_rca,region_col,product_col,time_col,value_col)
print('df_rca ready')

# add product codes back in 
# -- we need these to merge data from other sources, 
# -- we dont necessarily need the product names - strings - 
# -- but they are kept for illustrative purposes
# -- dft <- df_orig %>% select(product_name,product_code) %>% distinct()
# -- df_rca <- left_join(df_rca,dft,by=c('product_col'='product_name'))
colnames(df_rca)

# check results
sample_n(df_rca,10)

# Netherlands
df_rca %>% 
  filter(region_col=='Netherlands', time_col==2000) %>%
  arrange(desc(RCAcpt)) %>%
  select(product_col,RCAcpt) %>%
  head(n=5)

# Saudi 
df_rca %>% 
  filter(region_col=='Saudi Arabia', time_col==2000) %>%
  arrange(desc(RCAcpt)) %>%
  select(product_col,RCAcpt) %>%
  head(n=5)

# Mcp
df_rca$Mcp <- 0
df_rca[df_rca$RCAcpt>1, 'Mcp'] <- 1

# Product densities / Combinations
calc_cppt <- function(data,region_col,product_col) {
  # create product_col_2 column (to create all combinations of products within columns)
  #  rename and create prouct col 2 
  data$product_col_1 <- data$product_col
  data$product_col_2 <- data$product_col_1
  # create all product combinations within countries
  print('creating combinations')
  dft3 <- data %>% group_by(region_col) %>% tidyr::complete(product_col_1, product_col_2)
  print('combinations ready')
  # drop diagonal
  dft3 <- filter(dft3,product_col_1!=product_col_2)
  # calculate N of times that {product_col}s occur together
  dft3$count = 1
  dft3 <- dft3 %>% 
    group_by(product_col_1,product_col_2) %>% 
    summarise(Cpp = sum(count))
  # calculate ubiquity
  df_ub <- data %>%
    group_by(product_col) %>%
    summarize(Mcp = sum(Mcp))
  # Merge ubiquity into cpp matrix
  df_ub <- df_ub %>%
    rename(product_col_1 = product_col)
  dft3 <- left_join(dft3,df_ub,by=c('product_col_1'))
  df_ub <- df_ub %>%
    rename(product_col_2 = product_col_1)
  dft3 <- left_join(dft3,df_ub,by=c('product_col_2'))
  # Take minimum of conditional probabilities
  dft3$kpi = dft3$Cpp/dft3$Mcp.x
  dft3$kpj = dft3$Cpp/dft3$Mcp.y
  dft3$phi = dft3$kpi
  dft3 <- dft3 %>% 
      #mutate(phi = replace(phi, kpj < kpi, kpj))
      mutate(phi = ifelse(kpj < kpi, kpj ,kpi))
  ############
  return(dft3)
}

# Keep only year 1995
dft <- filter(df_rca,time_col==1995,Mcp==1)
df_cppt <- calc_cppt(data=dft,region_col,product_col)
print('df_cppt ready')

# Product that co-occur most often
df_cppt %>% 
  arrange(desc(Cpp)) %>%
  head(n=20)

# Most proximate products
df_cppt %>% 
  arrange(desc(phi)) %>%
  head(n=20)

# Patents 
# -- use 'foreign' library with read.dta function to read STATA file
dfp <- read.dta('https://www.dropbox.com/s/nwox3dznoupzm0q/patstat_year_country_tech_inventor_locations.dta?dl=1')

# Sample
sample_n(dfp,10)

# rename columns accordingly
dfp_rca <- dfp %>%
  rename(time_col = year,
         region_col = country_name,
         product_col = tech,
         value_col = count)
# calculate RCA
dfp_rca <- calc_rca(data=dfp_rca,region_col,product_col,time_col,value_col)

# What were Japan and Germany specialized in, in 1960 and 2010?
countries <- list("Japan", "Germany")
years <- list(1960, 2010)
for (country in countries) {
  for (year in years) {
    dft <- dfp_rca %>%
      filter(region_col == country, time_col == year) %>%
      arrange(desc(RCAcpt)) %>%
      head
    print(dft)
  }
}

# What technology classes are most proximate (in 2010)?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# Define Mcp
dfp_rca$Mcp = 0
dfp_rca <- dfp_rca %>% 
  mutate(Mcp = ifelse(RCAcpt> 1, 1,0))

# Keep only years 2010 and only country-product combinations where Mcp == 1 (thus RCAcp > 1)
dft <-  dfp_rca %>%
  filter(time_col==2010,Mcp==1)

# Calculate cppt
dfp_cppt <- calc_cppt(data=dft,region_col,product_col)
print('df_cppt ready')
print('cppt patent co-occurences and proximities dataframe ready')

# Show most proximate technologies
colnames(df_cppt)
dfp_cppt %>% 
  arrange(desc(phi)) %>% 
  head


# Product space
COUNTRY_STRING <- 'Saudi Arabia'
df_ps <- df_rca %>% 
  filter(region_col==COUNTRY_STRING)
# Cross-check
if (dim(df_ps)[1]==0) {
    print('Country string set above does not exist in data, typed correctly?')
    STOP
}


df_ps <- df_rca %>%
  filter(region_col==COUNTRY_STRING, time_col==2005,RCAcpt>1,value_col > 4000000) %>%
  select(product_col,value_col,product_code)
# Select year (2005), RCA > 1, minimum export value (4 million $)
# create product space now
# write to tempdir
# tempdir()
file_name <- tempfile(fileext = ".csv")
write.csv(df_ps, file=file_name,row.names=FALSE)
sprintf('saved in R temp folder as %s',file_name)
# save in home dir (non google co lab)
write.csv(df_ps, file='~/df_product_space.csv',row.names=FALSE)
print('df_ps ready to plot')

# Call python functions from R:
source_python('/Users/admin/Dropbox/proj/git_clones/py-productspace/create_product_space_v2.py')
source_python('https://raw.githubusercontent.com/cid-harvard/py-productspace/master/create_product_space_v2.py')

# Run python code directly but can't pass in R parameters this way
y = create_product_space(df_plot_dataframe_filename='~/df_product_space.csv',
                     df_plot_dataframe=df_ps,
                     df_plot_node_col='product_code',
                     df_node_size_col='value_col',
                     output_image_file ='/Users/admin/Dropbox/testnetwork.png'
                     )

# From temp file (use glue here to add R variables as Python parameter)
string_python <- glue("create_product_space(df_plot_dataframe_filename ='{file_name}', \\\
                     df_plot_node_col = 'product_code', \\\
                     df_node_size_col = 'value_col', \\\
                     output_image_file ='/Users/admin/Dropbox/testnetwork.png')"
                     )
py_run_string(string_python)
print('product space saved')

# Load image and show, using 'imager' library
network_img <- load.image('/Users/admin/Dropbox/testnetwork.png')
dim(network_img)
class(network_img)
plot(network_img)

# Economic complexity
# R Package by Carlo Bottai among others

https://cran.r-project.org/web/packages/economiccomplexity/index.html


# Examples
# -- example dataset in ecomplexity package
head(world_trade_avg_1998_to_2000,100)
dim(world_trade_avg_1998_to_2000)
typeof(world_trade_avg_1998_to_2000)
bi <- balassa_index(world_trade_avg_1998_to_2000)
head(bi)
typeof(bi)
com_fit <- complexity_measures(bi)
com_fit[0:5]
typeof(com_fit)
length(com_fit)
com_fit$complexity_index_country[1:5]
com_fit$complexity_index_product[1:5]
# convert to tibble dataframe
df_mtx <- com_fit$complexity_index_country %>% 
  as_tibble()

# apply to own trade data
# keep  year = 2000
dft <- df_orig %>% filter(year==2000) %>% select (country_name,product_name,export_value)
dft <- dft %>%
  rename(country = country_name,
         product = product_name,
         value = export_value)
# balassa index (rca, 1 if > 1)
bi <- balassa_index(dft)
# calculate eci / pci, using reflections here: same values as py-ecomplexity package
cm <- complexity_measures(bi,method='reflections')
# convert to tibble, add country names, sort from most to least complex
# -- xci labels are set with setNames (extract with 'names')
df_eci <- cm$complexity_index_country %>% 
  as_tibble() %>%
  mutate(country = names(cm$complexity_index_country)) %>%
  rename(eci = value)
# same procedure for products (pci)
df_pci <- cm$complexity_index_product %>% 
  as_tibble() %>%
  mutate(product = names(cm$complexity_index_product)) %>%
  rename(pci= value)
# add product codes as well to df_pci 
df_product_codes <- df_orig %>%
  select(product_name,product_code) %>%
  distinct(product_name, .keep_all= TRUE) # drop_duplicates in pandsa
df_pci <- left_join(df_pci,df_product_codes,by=c('product'='product_name'))
# inspect most complex countries
df_eci %>% arrange(desc(eci)) %>% head
# inspect most complex products
df_pci %>% arrange(desc(pci)) %>% head

# Complexity by destination
print('loading data (from dropbox)')
df_ukr <- read_csv(file='~/Dropbox/proj/org_zhtml_projects/product-space-eci-workshop/files/ukr_exports_per_destination.csv')
# df_ukr <- read_csv(file='https://www.dropbox.com/s/megm8qzn3jcwnqz/ukr_exports_per_destination.csv?dl=1')
print('loaded')

# show sample of dataset
sample_n(df_ukr,10)

# merge pcis from 2000 into dataframe
# -- add leading zero if 3-digit
df_ukr$len_hs_product_code = str_length(df_ukr$hs_product_code) 
df_ukr <- df_ukr %>% 
    mutate(hs_product_code = ifelse(len_hs_product_code == 3, paste('0',hs_product_code,sep=""),hs_product_code))
# -- strip leading / trailing spaces
df_ukr$hs_product_code <- trimws(df_ukr$hs_product_code, which = c("both"))
df_ukr <- left_join(df_ukr, df_pci, by = c("hs_product_code" = "product_code"))

calc_ecimc <- function(data,origin_col,destination_col,product_col,value_col,pci_col) {
           dft <- data
           #####
           dft <- dft %>% 
             group_by(origin_col,destination_col) %>% 
             mutate(export_value_cot = sum(value_col)) 
           ##### 
           dft$pci_x_export = dft$pci_col * dft$value_col
           #####
           dft <- dft %>% 
             group_by(origin_col,destination_col) %>% 
             mutate(pci_x_export_sum = sum(pci_x_export)) 
           #####
           dft$eciMc = dft$pci_x_export_sum / dft$export_value_cot
           #####
           # dft <- dft %>%
           #   distinct(c('origin_col','destination_col'), .keep_all= TRUE) 
           dft <- dft %>%
             distinct(origin_col,destination_col, .keep_all= TRUE) 
           #####
           dft <- dft %>% select(origin_col,destination_col,eciMc)
           ###################
           return(dft)
         }
# rename columns accordingly
colnames(df_ukr)
df_ukr_ecimc <- df_ukr %>%
  rename(origin_col = location_code,
         destination_col = partner_code,
         product_col = hs_product_code,
         value_col = export_value,
         pci_col = pci
  )
# calculate ecimc
df_ukr_ecimc <- calc_ecimc(data=df_ukr_ecimc,origin_col,destination_col,product_col,value_col,pci_col)
head(df_ukr_ecimc)

# Most complex exports to where?
head(arrange(df_ukr_ecimc,desc(eciMc)))

# map it
df_ukr_ecimc_map <- df_ukr_ecimc %>%
  rename("iso-a3"="destination_col")

hcmap(
  map = "custom/world-highres3", # high resolution world map
  # eci in 85th percentile
  data = filter(df_ukr_ecimc_map,eciMc>quantile(df_ukr_ecimc_map$eciMc,probs=c(0.85),na.rm=TRUE)),
  joinBy = "iso-a3",
  value = "eciMc",
  showInLegend = FALSE, # hide legend
  #nullColor = "#DADADA",
  download_map_data = TRUE
) %>% hc_colorAxis(minColor = "orange", maxColor = "red")

# Excercise answers

*** What product does Ukraine export most in 1995? (excluding services such as 'transport', 'ict' etc)

df_orig %>% 
  filter(country_name=='Ukraine',year==2005) %>%
  group_by(product_code,product_name) %>% 
  summarise(sum_export_value = sum(export_value)) %>%
  arrange(desc(sum_export_value)) %>%
  head()

*** What products is Ukraine specialized in 1995 and 2005 and how much do they export of these?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# 1995

df_rca %>% 
  filter(region_col=='Ukraine',time_col==1995) %>%
  arrange(desc(RCAcpt)) %>%
  select(product_col,RCAcpt,value_col) %>%
  head

df_rca %>% 
  filter(region_col=='Ukraine',time_col==2005) %>%
  arrange(desc(RCAcpt)) %>%
  select(product_col,RCAcpt,value_col) %>%
  head


*** Which product is most related to the product 'Stainless steel wire'?

df_cppt %>% 
  filter(product_col_1=='Stainless steel wire') %>%
  arrange(desc(phi)) %>%
  head()

# *** Plot Ukraine in the product space in 1995.

# How would you characterize Ukraine's position in the product space?

df_ps <- df_rca %>%
  filter(region_col=='Ukraine', time_col==1995,RCAcpt>1,value_col > 4000000) %>%
  select(product_col,value_col,product_code)
# Select year (2005), RCA > 1, minimum export value (4 million $)
# create product space now
# write to tempdir
# tempdir()
file_name <- tempfile(fileext = ".csv")
write.csv(df_ps, file=file_name,row.names=FALSE)
sprintf('saved in R temp folder as %s',file_name)
# save in home dir (non google co lab)
write.csv(df_ps, file='~/df_product_space.csv',row.names=FALSE)

# Call python functions from R:
source_python('https://raw.githubusercontent.com/cid-harvard/py-productspace/master/create_product_space_v2.py')
source_python('/Users/admin/Dropbox/proj/git_clones/py-productspace/create_product_space_v2.py')

# From temp file (use glue here to add R variables as Python parameter)
string_python <- glue("create_product_space(df_plot_dataframe_filename ='{file_name}', \\\
                     df_plot_node_col = 'product_code', \\\
                     df_node_size_col = 'value_col', \\\
                     output_image_file ='/Users/admin/Dropbox/testnetwork.png')"
                     )
py_run_string(string_python)
print('product space saved')
# Load image and show, using 'imager' library
network_img <- load.image('/Users/admin/Dropbox/testnetwork.png')
plot(network_img)

*** Plot Ukraine in the product space in 2015.
Do you notice a difference with 1995?

df_ps <- df_rca %>%
  filter(region_col=='Ukraine', time_col==2015,RCAcpt>1,value_col > 4000000) %>%
  select(product_col,value_col,product_code)
# Select year (2005), RCA > 1, minimum export value (4 million $)
# create product space now
# write to tempdir
# tempdir()
file_name <- tempfile(fileext = ".csv")
write.csv(df_ps, file=file_name,row.names=FALSE)
sprintf('saved in R temp folder as %s',file_name)
# save in home dir (non google co lab)
write.csv(df_ps, file='~/df_product_space.csv',row.names=FALSE)

# Call python functions from R:
source_python('https://raw.githubusercontent.com/cid-harvard/py-productspace/master/create_product_space_v2.py')
source_python('/Users/admin/Dropbox/proj/git_clones/py-productspace/create_product_space_v2.py')

# From temp file (use glue here to add R variables as Python parameter)
string_python <- glue("create_product_space(df_plot_dataframe_filename ='{file_name}', \\\
                     df_plot_node_col = 'product_code', \\\
                     df_node_size_col = 'value_col', \\\
                     output_image_file ='/Users/admin/Dropbox/testnetwork.png')"
                     )
py_run_string(string_python)
print('product space saved')
# Load image and show, using 'imager' library
network_img <- load.image('/Users/admin/Dropbox/testnetwork.png')
plot(network_img)

*** Plot your own country across different years in the product space. Do the results make sense? Do you notice any patterns?

** Excercise 2:

*** What are countries with high complexity in 2000?

df_eci %>%
  arrange(desc(eci)) %>%
  head

*** Vice versa, what are countries with low complexity in 2015?

df_eci %>%
  arrange(desc(eci)) %>%
  tail()

*** What are products (PCI) with high complexity in 2015?


df_pci %>%
  arrange(desc(pci)) %>%
  head()

*** Vice versa, what are products (PCI) with low complexity in 2015?

df_pci %>%
  arrange(desc(pci)) %>%
  tail()

*** Ukraine

**** How did Ukraine\'s economic complexity evolve over time?

df_eci_allyrs <- data.frame()
years <- 1995:2016
  for (yeart in years) {
    sprintf('doing year %s',yeart)
    # Loop now
    head(df_orig)
    dft <- df_orig %>% filter(year==yeart) %>% select (country_name,product_name,export_value)
    dft <- dft %>%
      rename(country = country_name,
             product = product_name,
             value = export_value)
    # balassa index (rca, 1 if > 1)
    bi <- balassa_index(dft)
    # calculate eci / pci, using reflections here: same values as py-ecomplexity package
    cm <- complexity_measures(bi,method='reflections')
    # convert to tibble, add country names, sort from most to least complex
    # -- xci labels are set with setNames (extract with 'names')
    df_eci <- cm$complexity_index_country %>% 
      as_tibble() %>%
      mutate(country = names(cm$complexity_index_country)) %>%
      rename(eci = value)
    df_eci$year = yeart
    head(df_eci)
    df_eci_allyrs <- bind_rows(df_eci_allyrs,df_eci)
  }

# plot eci over the years
dft <- filter(df_eci_allyrs,country == 'Ukraine')
dft %>% ggplot(aes(x = year, y = eci)) + geom_line()

#+END_SRC

**** How does Ukraine\'s economic complexity in 2015 compare to other countries? Which countries have comparable economic complexity?

# - keep 2015, sort by eci and create row number
dft <- df_eci_allyrs %>%
  filter(year == 2015)  %>%
  arrange(desc(eci)) %>%
  mutate(row_number = row_number())
# countries above and below Ukraine in ranking
row_number1 <- filter(dft,country=='Ukraine')$row_number-5
row_number2 <- filter(dft,country=='Ukraine')$row_number+5
# show 
dft %>% slice(row_number1:row_number2)


**** What are the most complex products that Ukraine exported with rca > 1 in 2015?

df_ukr <- df_rca %>%
  filter(time_col == 2015,region_col=='Ukraine',RCAcpt > 1)
# to merge pci into the dataframe, add leading zero if 3-digit
df_ukr$len_product_code = str_length(df_ukr$product_code) 
df_ukr <- df_ukr %>% 
    mutate(product_code = ifelse(len_product_code == 3, paste('0',product_code,sep=""),product_code))
# strip leading / trailing spaces
df_ukr$product_code <- trimws(df_ukr$product_code, which = c("both"))
# merge pcis from 2000 into dataframe
df_ukr <- left_join(df_ukr, df_pci, by = c("product_code" = "product_code"))
# sort
head(arrange(df_ukr,desc(pci)))
head(df_ukr,20)

----------------------------

https://rstudio.github.io/reticulate/articles/calling_python.html

Built in conversion for many Python object types is provided, including NumPy arrays and Pandas data frames. For example, you can use Pandas to read and manipulate data then easily plot the Pandas data frame using ggplot2:

y
y = create_product_spaces()

This works in R Markdown in R Studio:
```{python}
import numpy as np
my_python_array = np.array([2,4,6,8])
for item in my_python_array:
    print(item)
```
```{python}
import requests
url = 'https://raw.githubusercontent.com/cid-harvard/py-productspace/master/create_product_space_v2.py'
r = requests.get(url)
exec(r.content)
print('product space code imported into python')
```

See this!

Convert between Python and R

Much code at Growth Lab written in Python
So majority in R, still incorporate pieces into Python

e.g. programming capabilities of Python with statistical capabilities of R

!!!!!!!!!

https://semba-blog.netlify.app/07/10/2020/integrate-r-and-python-in-rstudio-to-power-your-analytical-capability/

PRODUCT SPACE: LOAD PYTHON LIBRARY IN R!
reticulate
https://cran.r-project.org/web/packages/reticulate/vignettes/calling_python.html

Load package from github rather than function


A great way to generate network graphs is to combine functions from the igraph, the ggraph, and the tidygraph packages. The advantages are that the syntax of for creating the networks aligns with the tidyverse style of writing R and that the graphs can be modified very easily.


- You're running into issues with the tidyverse and what is referred to as non-standard evaluation (NSE)
    # Most tidyverse stuff uses non-standard eval so symbols are evaluated in the context of the data frame or tibble that's being used."
- Good post: https://www.brodieg.com/2020/05/05/on-nse/
- Painful with !!sym and many functions wont work properly
- I usually use dplyr, 
but I also have projects where the semantic meaning of columns is 
the same regardless of their name.  So I frequently 
do a renaming of columns to "id, role, time, measure, value" 
even though the data often comes in with columns like "provider ID, service unit, time of shift, name, reading" (edited)

Message "groupby"

country_col = 'country_name'
time_col = 'year'
value_col = 'export_value'
df_rca2 <- df_rca %>% 
  group_by(!!sym(country_col),!!sym(time_col)) %>% 
  mutate(Xct = sum(!!sym(value_col))) 

