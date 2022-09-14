#+OPTIONS: broken-links:t
#+PROPERTY: header-args :session "~/Dropbox/local.json" :eval never-export :exports results
#+TITLE: Trade and patents in Python: RCAs, proximities, product space and economic complexity
#+OPTIONS: toc:nil
#+OPTIONS: author:nil
#+OPTIONS: date:nil

September 16 2022, Matte Hartog

*  Notes

Google colab link:

https://colab.research.google.com/github/matteha/product-space-eci-workshop/blob/main/product-space-eci-workshop-2022.ipynb

Google colab link in R:
https://colab.research.google.com/github/matteha/product-space-eci-workshop/blob/main/product-space-eci-workshop-in-R.ipynb

To run code in Google Colab, you need a Google account.

** Other :noexport:

C-= to surround blocks in SRC_python

DONT ADD :noexport: TO MAIN HEADINGS SINCE it'll give a 'killed buffer' error
Dont add :noexport to main heading, then cant export to ipynb, 'killed buffer' error.

Copy / paste in Google Colab with shift + control + c and shift + control + v

Use  BEGIN_EXAMPLE to deal with underscores and have them visualized properly in Google Colab, e.g.:

#+BEGIN_EXAMPLE
"country_partner_hsproduct4digit_years_2000_2016.csv"
#+END_EXAMPLE

+ Export org file to jupyter notebook

config file: (defun ox-org-export-to-ipynb-jupyter-push-github ()

  - ox-ipynb: https://github.com/jkitchin/ox-ipynb

Use \nonumber in equations to not number them

Use (update-tag) to update numbering

((neotree-copy-filepath-to-yank-ring))
((org-latex-preview))
and
((org-latex-preview-all-buffer))


     - Ukraine report at https://growthlab.cid.harvard.edu/files/growthlab/files/2020-10-cid-fellows-wp-129-ukraine-role.pdf

       [[~/Dropbox/2020-10-cid-fellows-wp-129-ukraine-role.pdf]]

Ukraine report, latex:
[[/home/linux/Dropbox/proj/prog/Overleaf/Ukraine_WP_2020/main.tex]]


Ukraine report, latex pdf:
[[/home/linux/Dropbox/proj/prog/Overleaf/Ukraine_WP_2020/main.pdf]]

Product space paper:

[[~/Dropbox/product_space_2007.pdf]]

Product space Supplementary Material / compendium appendix:

[[~/Dropbox/Hidalgo.SOM.pdf]]

Next week Wednesday:
- Summer school: check out google colab!!!
- *35 Students*
  Prepare: short deck of slides as to how long deck would look like
  Session 3: Matte (Full notebook of what you have to do, remove some steps that they have to figure out themselves?)
    - Gravity models Ukraine?
  Session 4: Matte, first 45 minutes, collect ideas for this
- Can we run all packages / complexity there?
  - UMAP?
- STATA kernel?
  - Enough memory? Either on jupyterhub or student computers?
  - Diversification, diffusion, coordination
  - Can everyone run all packages? Student pcs / cloud?
    SEE site:
    https://jupyter4edu.github.io/jupyter-edu-book/getting-going.html
  - Spring 2022 module: methods in economic complexity at Harvard, 6/7 weeks

  - JupyterHub? Harvard?
    harvard JupyterHub hks

    JupyterHub is open-source software that provides a cloud-based Jupyter application for each user in a group. Each user has their own account and home directory on the server. The Hub, JupyterHub’s central system, allows authenticating users and starting individual Jupyter notebook servers. Programs that start notebook servers can use a variety of technical solutions. For more details, see https://github.com/jupyterhub/jupyterhub/wiki/Spawners

    Brown university:

    https://ithelp.brown.edu/kb/articles/jupyterhub-overview

  https://docs.google.com/forms/d/e/1FAIpQLSe_Ys183PSqoaU-WyyS0dYG3fSWo_kbjOnaAFRsNuEzGcgL6Q/viewform

  FAS OnDemand Jupyter Notebooks Request Form

This form is for gathering information needed to set up an FAS OnDemand Jupyter Notebooks installation for FAS/SEAS courses.

The anticipated turnaround time for requests is two business days. If a request will take longer than that to complete, a representative from Academic Technology for FAS will be in touch with the contact person listed on the form to provide an update.

3) Use a server if you can (JupyterHub, Sage Math Cloud, etc.) This can be a form of equity to make sure that everyone has the same computing resources.

***  Communication

Hi all,

Tomorrow during the Review Session we will be using Python to calculate and explore Xcps/RCAs/Mcps and proximities in trade and patents, visualize trade data in the product space, calculate ECIs/PCIs and ECIs weighted by destination (focusing on Ukraine) and do very basic geovisualization.

As we'll be using Google Colab, please create a Google account if you don't have one. An account is necessary to run code on Colab and to save changes you make to the Jupyter notebook (on Google Drive or Github).

The Jupyter notebook that we'll be using tomorrow can be found at:

https://colab.research.google.com/github/matteha/product-space-eci-workshop/blob/main/product-space-eci-workshop.ipynb

See you then!

Matte

* To do first

In Google Colab:

1. Turn on Table of Contents: (in browser, click on 'View' in top, then 'Table of Contents')

2. Expand all sections ('View' > 'Expand Sections' if not greyed out)


* Outline of lab session


Google Colab gives you access to a Jupyter Notebook with a Python kernel (but other kernels are also possible, e.g. R, see

https://github.com/IRkernel/IRkernel

)

We'll cover:

- Introduction to trade data

- Calculating RCAs, co-occurences and proximities
  - Product and technology proximities
  - Trade: specialization of countries over time
  - Technologies (patents): specialization of countries and US cities over time

- Product space visualization

- Calculating Product Complexity / Economic Complexity (by destination)


* Trade data

** Background

The product space is based on trade data between countries (as well as its derivations / related measures such as economic complexity and the Growth's annual rankings of countries by economic complexity (at [[https://atlas.cid.harvard.edu]])).

The Growth Lab maintains and periodically updates a cleaned version of trade data at Harvard Dataverse:

https://dataverse.harvard.edu/dataverse/atlas

This dataset contains bilateral trade data among 235 countries and territories in thousands of different products categories (a description of the data can be found at: http://atlas.cid.harvard.edu/downloads).

How does the data look like? We will explore the data in Python using the 'pandas' (most popular Python package for data analysis) - what 'tidyverse' is to R.

*** Footnote on trade and services (ICT, tourism, etc.):
- Services and tourism are included in the Growth Lab's Atlas and trade data as well as of September 2018. See announcement at:

https://atlas.cid.harvard.edu/announcements/2018/services-press-release

Obtained from IMF, trade in services covers four categories of economic activities between producers and consumers across borders:

- services supplied from one country to another (e.g. call centers)
- consumption in other countries (e.g. international tourism)
- firms with branches in other countries (e.g. bank branches overseas)
- individuals supplying services in another country (e.g. IT consultant abroad)

** Load necessary Python libraries

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# -- Global settings
# - import python libraries necessary for this workshop
# suppress warnings on google colab for now
import warnings
warnings.filterwarnings("ignore")
# to interact with os, e.g. to execute shell comands such as 'ls', 'pwd' etc.
import os
# to do data processing
import pandas as pd
# backend of pandas, working with matrices
import numpy as np
# to visualize data (in pandas)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
# to process a json file
import json
# work with regex in python
import re
# work with networks in python, to create product space
import networkx as nx
# python tools to work with combinations of arrays
from itertools import count
from itertools import combinations
from itertools import product
# to download files
import urllib.request, json
# to plot data on maps
# pip install is needed on Google Colab
!pip3 install geopandas
import geopandas
# -- set scientific notation to display numbers fully rather than exponential
pd.set_option('display.float_format', '{:.2f}'.format)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all' # Show all results of jupyter
import seaborn as sns
sns.set_style('whitegrid') # Display grids on dark background
# to run regressions
import statsmodels.api as sm
# Enlarged pandas display - more colums and rows with greater width
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth',300)
print('necessary libraries loaded')
#+END_SRC

** Download trade dataset and load into memory

Trade data is constantly updated by the Growth Lab, you can find the most recent version of the trade data at our Dataverse here:

https://dataverse.harvard.edu/dataverse/atlas

Below we're using the trade data using the HS classification ('Harmonized System 1992' - alternative is 'SITC - Standard Industrial Trade Classification' which goes back further in time) at the 4 digit level (alternative is 2 or 6 - 6 has more detail). It can be found here:

https://dataverse.harvard.edu/file.xhtml?fileId=4946953&version=4.0

The trade file we're using below has a fix implemented in the labels (strings) of the products - some products currently erronuously have the same strings (e.g. product codes 5209 and 5211 in Zimbabwe have the same product string). The file is regularly updated by the Growth Lab.

To load the data directly we're using a Dropbox link with the fix implemented (Dataverse normally requires one to fill in an agreement form first but see Section 1 (by Shreyas) on how to use the dataverse library in R to load data from there directly into R).

(The trade file is a large file because it includes country / product strings, takes 1 - 3 minutes to load. One can also merge strings in separately but for illustrative purposes it is easier to have them all preloaded - also to avoid memory problems that R quickly runs into).

Our $X_{cpt}$ matrix.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes

# Load the necessary data into pandas
df_orig = pd.read_csv('https://www.dropbox.com/s/3n4r4qo4j0jjpln/trade.csv?dl=1')
print('trade data loaded')
#+END_SRC


** Save locally ::noexport::
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# Save locally
#df_orig.to_csv(f'~/Dropbox/trade.csv',index=False, encoding='utf-8')
#sprint('saved')
dft = df_orig.copy()
dft.drop(['country_name'],axis=1,inplace=True,errors='ignore')
dft.drop(['product_name'],axis=1,inplace=True,errors='ignore')
dft_country_strings = df_orig[['country_code','country_name']].drop_duplicates()
dft_product_strings = df_orig[['product_code','product_name']].drop_duplicates()
dft.to_csv(f'~/Dropbox/trade_without_labels.csv',index=False, encoding='utf-8')
dft.to_parquet(f'~/Dropbox/trade_without_labels.parquet')
dft_country_strings.to_csv(f'~/Dropbox/trade_country_labels.csv',index=False, encoding='utf-8')
dft_product_strings.to_csv(f'~/Dropbox/trade_product_labels.csv',index=False, encoding='utf-8')
print('saved')
#+END_SRC

** Exploring the trade data

*** Structure of dataset
Our $X_{cpt}$ matrix:
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
df_orig.sample(n=5) # show 5 random rows
#+END_SRC

*** What years are in the data?
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
df_orig['year'].unique()
#+END_SRC

*** How many products are in the data?
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
df_orig['product_name'].nunique()
#+END_SRC

*** Finding specific countries / products based on partial string matching

If you're interested in finding data on certain countries / products but not sure how exactly these are spelled in the data (or are spelled with / without e.g. capital letters).

# To find the exact naming of a country / product, use str.contains on STRING:
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
STRING = 'Netherland'
df_orig[df_orig['country_name'].str.contains(STRING)][['country_name']].drop_duplicates()
#+END_SRC

You can also include regex expressions here, e.g. to ignore lower/uppercase ('wine' vs 'Wine')

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
STRING = 'wine'
df_orig[df_orig['product_name'].str.contains(STRING,flags=re.IGNORECASE, regex=True)][['product_name']].drop_duplicates()
#+END_SRC

*** Example: What were the major export products of the USA in 2012?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# create a 'dataframe' called 'df2' with only exports from USA in 2012
df2 = df_orig[ (df_orig['country_code']=='USA') & (df_orig['year'] == 2012) ].copy()
# create another dataframe 'df3' that contains the sum of exports per product
df3 = df2.groupby(['product_code','product_name'],as_index=False)['export_value'].sum()
# sort
df3.sort_values(by=['export_value'],ascending=False,inplace=True)
# show first 10 rows
df3[0:10]
#+END_SRC

*** Example: How did exports of Cars evolve over time in the USA?
From about 10 billion USD up to almost $60 billion USD.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
df2 = df_orig[ (df_orig['country_code']=='USA')].copy()
#df3 = df2[df2['product_name']=='Cars']
df3 = df2[df2['product_code']=='8703']
df3.plot(x='year', y='export_value')
#+END_SRC


* Revealed comparative advantage (RCA)
What products are countries specialized in? For that, following Hidalgo et al. (2007), we calculate the Revealed Comparative Advantage (RCA) of each country-product pair: how much a country 'over-exports' a product in comparison to all other countries.

Technically this is the Balassa index of comparative advantage, calculated as follows for product $p$ and country $c$ at time $t$:

\begin{equation} \label{e_RCA}
{RCA}_{cpt}=\frac{X_{cpt}/X_{ct}}{X_{pt}/X_{t}}
\tag{1}
\end{equation}

where $X_{cpt}$ represents the total value of country $c$’s exports of product $p$ at time $t$ across all importers. An omitted subscript indicates a summation over the omitted dimension, e.g.: $X_{t}=\sum \limits_{c,p,t} X_{cpt}$.

A product-country pair with $RCA>1$ means that the product is over-represented in the country's export basket.

We use the original trade dataset ('df_orig') that is loaded into memory, calculating RCAs as follows:

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
def calc_rca(data,country_col,product_col,time_col,value_col):
    """
    Calculates Revealed Comparative Advantage (RCA) of country-product-time combinations

    Returns:
        pandas dataframe with RCAs
    """

    # Aggregate to country-product-time dataframe
    print('creating all country-product-time combinations')
    # - add all possible products for each country with export value 0
    # - else matrices later on will have missing values in them, complicating calculations
    df_all = pd.DataFrame(list(product(data[time_col].unique(), data[country_col].unique(),data[product_col].unique())))
    df_all.columns=[time_col,country_col,product_col]
    print('merging data in')
    df_all = pd.merge(df_all,data[[time_col,country_col,product_col,value_col]],how='left',on=[time_col,country_col,product_col])
    df_all.loc[df_all[value_col].isnull(),value_col] = 0

    # Calculate the properties
    print('calculating properties')
    df_all['Xcpt'] = df_all[value_col]
    df_all['Xct'] = df_all.groupby([country_col, time_col])[value_col].transform(sum)
    df_all['Xpt'] = df_all.groupby([product_col, time_col])[value_col].transform(sum)
    df_all['Xt'] = df_all.groupby([time_col])[value_col].transform('sum')
    df_all['RCAcpt'] = (df_all['Xcpt']/df_all['Xct'])/(df_all['Xpt']/df_all['Xt'])
    # set to 0 if missing, e.g. if product / country have 0 (total) exports
    df_all.loc[df_all['RCAcpt'].isnull(),'RCAcpt'] = 0
    # drop the properties 
    df_all.drop(['Xcpt','Xct','Xpt','Xt'],axis=1,inplace=True,errors='ignore')

    return df_all

df_rca = calc_rca(data=df_orig,country_col='country_name',product_col='product_name',time_col='year',value_col='export_value')

print('rca dataframe ready')

# Add product codes back in (we need these to merge data from other sources, we dont necessarily need product names but they are kept for illustrative purposes)
df_rca = pd.merge(df_rca,df_orig[['product_name','product_code']].drop_duplicates(),how='left',on='product_name')

print('df_rca ready')
# show results

#+END_SRC

Note: You can operate (more) directly on matrices with NumPy (Pandas is built on NumPy), e.g. to calculate RCA:

#+BEGIN_EXAMPLE
Xc = np.sum(Xcp, axis=1)
Xp = np.sum(Xcp, axis=0)
X = np.sum(Xcp)
RAC_mat = np.copy(Xcp)
for i in range(len(RAC_mat)):
    RAC_mat[i, :] = RAC_mat[i, :] / Xc[i]
# products / countries with no exports
RAC_mat[np.isnan(RAC_mat)] = 0
RAC_mat = np.transpose(RAC_mat)
for i in range(len(RAC_mat)):
    RAC_mat[i, :] = RAC_mat[i, :] / (Xp[i] / X)
#+END_EXAMPLE

** Sample of dataset
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
df_rca.sample(n=5)
#+END_SRC

** Example: What products are The Netherlands and Saudi Arabia specialized in, in 2000?

(Note that different commands are chained together here; can also be ran separately)

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# The Netherlands
print("\n The Netherlands: \n")

df_rca[ (df_rca['year']==2000) & (df_rca['country_name']=='Netherlands')].sort_values(by=['RCAcpt'],ascending=False)[['product_name','RCAcpt','year']][0:5]

print("\n Saudi Arabia:\n")

# Saudi Arabia
df_rca[ (df_rca['year']==2000) & (df_rca['country_name']=='Saudi Arabia')].sort_values(by=['RCAcpt'],ascending=False)[['product_name','RCAcpt','year']][0:5]
#+END_SRC

* Product proximity (based on co-occurences)

** Calculating product co-occurences and proximities
Knowing which countries are specialized in which products, the next step analyzes the extent to which two products are over-represented ($RCA>1$) in the same countries.

As noted in the lecture, the main insight supporting this inference is that countries will produce combinations of products that require similar capabilities.

Hence we infer capabilities from trade patterns, because the capabilities of a country is a priori hard to determine and capabilities themselves are hard to observe.

Hence, *the degree to which two products cooccur in the export baskets of the same countries provides an indication of how similar the capability requirements of the two products are*.

We will calculate the co-occurence matrix of products below.

First, a product is 'present' in a country if the country exports the product with $RCA>1$:
    
\begin{equation} \label{e_presence}
M_{cp}=\begin{cases}
    1 & \text{if ${RCA}_{cp}>1$}; \\
    0 & \text{elsewhere.}
    \end{cases}
\tag{2}
\end{equation}


#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
df_rca['Mcp'] = 0
df_rca.loc[df_rca['RCAcpt']>1,'Mcp'] = 1
#+END_SRC

Next, we calculate how often two products are present in the same countries, using the Mcp threshold:
    
\begin{equation} \label{e_cooc}
C_{pp'}=\sum \limits_{c} M_{cp} M_{cp'}
\tag{3}
\end{equation}

To get an accurate value of product proximity, we need to correct these numbers for the extent to which products are present in general in trade flows between countries. To do so, Hidalgo et al. (2007) calculate product proximity as follows, defining it as the minimum of two conditional probabilities:

\begin{equation}
C_{ppt'}  = \min \left( \frac{C_{pp'}}{C_{p}},\frac{C_{pp'}}{C_{p'}} \right)
\tag{4}
\end{equation}

#+BEGIN_COMMENT
To do so, Hidalgo et al. (2007) calculate product proximity as follows, defining $\tilde{\pi}_{pp'}$ as the minimum of two conditional probabilities: $\tilde{\pi}_{pp'} = \min \left( \frac{C_{pp'}}{C_{p}},\frac{C_{pp'}}{C_{p'}} \right)$.
#+END_COMMENT

The minimum here is used to elimate a 'false positive'.

#+BEGIN_COMMENT
$\phi_{ij} = \frac{\sum_c \{ M_{cp_i} * M_{cp_j} \}}{max \{k_{p_i}, k_{p_j}\}}$
#+END_COMMENT

Hence we correct for how prevalent specialization in product $i$ and product $j$ is across countries (i.e. the 'ubiquity' of the products).

We will use the first year of data in the dataset, *1995,* below.

Note that to reduce yearly votality, Hidalgo et al. (2007) aggregate the trade data across multiple years (1998-2000) when calculating RCAs and product proximities for the product space. (However, when comparing the product space across years, they do use individual years).

Function to calculate cppt:

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
def calc_cppt(data,country_col,product_col):
    """
    Calculates product co-occurences in countries

    Returns:
        pandas dataframe with co-occurence value for each product pair
    """

    # Create combinations within country_col (i.e. countries) of entities (i.e. products)
    dft = (data.groupby(country_col)[product_col].apply(lambda x: pd.DataFrame(list(combinations(x,2))))
            .reset_index(level=1, drop=True)
            .reset_index())
    dft.rename(columns={0:f'{product_col}_1'}, inplace=True)
    dft.rename(columns={1:f'{product_col}_2'}, inplace=True)

    # Create second half of matrix (assymmetrical):
    # -- {product_col} 1 X {product_col} 2 == {product_col} 2 X {product_col} 1
    dft2 = dft.copy()
    dft2.rename(columns={f'{product_col}_1':f'{product_col}_2t'}, inplace=True)
    dft2.rename(columns={f'{product_col}_2':f'{product_col}_1'}, inplace=True)
    dft2.rename(columns={f'{product_col}_2t':f'{product_col}_2'}, inplace=True)
    # -- add second half
    dft3 = pd.concat([dft,dft2],axis=0,sort=False)

    # Drop diagonal if present
    dft3 = dft3[ dft3[f'{product_col}_1'] != dft3[f'{product_col}_2'] ]

    # Now calculate N of times that {product_col}s occur together
    dft3['count'] = 1
    dft3 = dft3.groupby([f'{product_col}_1',f'{product_col}_2'],as_index=False)['count'].sum()
    dft3.rename(columns={f'count':f'Cpp'}, inplace=True)

    # Calculate ubiquity
    df_ub = data.groupby(product_col,as_index=False)['Mcp'].sum()
    # Merge ubiqity into cpp matrix
    df_ub.rename(columns={f'{product_col}':f'{product_col}_1'}, inplace=True)
    dft3 = pd.merge(dft3,df_ub,how='left',on=f'{product_col}_1')
    df_ub.rename(columns={f'{product_col}_1':f'{product_col}_2'}, inplace=True)
    dft3 = pd.merge(dft3,df_ub,how='left',on=f'{product_col}_2')

    # Take minimum of conditional probabilities
    dft3['kpi'] = dft3['Cpp']/dft3['Mcp_x']
    dft3['kpj'] = dft3['Cpp']/dft3['Mcp_y']
    dft3['phi'] = dft3['kpi']
    dft3.loc[dft3['kpj']<dft3['kpi'],'phi'] = dft3['kpj']

    return dft3

#+END_SRC


#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# Keep only year 1995
dft = df_rca[df_rca['year']==1995].copy()

# Keep only country-product combinations where Mcp == 1 (thus RCAcp > 1)
dft = dft[dft['Mcp']==1]

# Calculate cpp
df_cppt = calc_cppt(dft,country_col='country_name',product_col='product_name')

print('cppt product co-occurences and proximities dataframe ready')
#+END_SRC


*** Products that co-occur most often

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
df_cppt.sort_values(by=['Cpp'],ascending=False)[0:10]
#+END_SRC

*** Most proximate products
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" file :async yes
df_cppt.sort_values(by=['phi'],ascending=False)[0:10]
#+END_SRC

*** Cross-check product proximity Shreyas :noexport:

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# Cross-check with Shreyas package
from ecomplexity import proximity
from ecomplexity import density
#trade_cols = {'time':'year', 'loc':'country_code', 'prod':'product_code', 'val':'export_value'}
dft = df_rca[df_rca['year']==1995].copy()
#dft.rename(columns={f'product_name':f'product'}, inplace=True)
trade_cols = {'time':'year', 'loc':'country_name', 'prod':'product_name', 'val':'export_value'}
df_condp2 = proximity(dft, trade_cols)
df_condp2.rename(columns={f'proximity':f'phi'}, inplace=True)
df_condp2.rename(columns={f'product_name_1':f'product_1'}, inplace=True)
df_condp2.rename(columns={f'product_name_2':f'product_2'}, inplace=True)

df_C = pd.merge(df_cppt,df_condp2,how='left',on=[f'product_1','product_2'],indicator=True)
df_C['_merge'].value_counts()
df_C.sample(n=20)
#+END_SRC


** Side note: Normalize product co-occurences (cpp) as in Neffke 2017 :noexport:
The co-occurrences can also be normalized slightly differently, by the number of times we would expect products $p$ and $p'$ to co-occur, had co-occurrences taken place at random. $\tilde{\pi}_{pp'}$ is defined as the ratio of observed to expected co-occurrences (following Neffke 2017):

\begin{equation} \label{e_prox}
\tilde{\pi}_{ppt'} = \frac{C_{ppt'}}{C_{pt} C_{pt'}/C_{t}}
\tag{4}
\end{equation}

\begin{equation} \label{e_prox}
Cr_{ppt'} = \frac{C_{ppt'}}{C_{pt} C_{pt'}/C_{t}}
\tag{5}
\end{equation}

(note that this normalization is similar to how the RCA is calculated above, taking into account the trade volumes of each product and country).

Thus, in the end we measure the extent to which a country is an effective exporter (RCA > 1) of a given good /i/ given that the country has comparative advantage in good /j/.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# Keep only country-product combinations where Mcp == 1

def calc_crtp(data,cols_products,col_cpp,symmetrize=True):
    """
    Calculate normalized product proximity
    ###
    Returns:
        pandas dataframe with standardized proximity values for each product pair
    """

    if symmetrize == True:
        # We symmetrize the matrix first by appending the 'second half' of the matrix to the dataset
        # - needed because the co-occurence is symmetric rather than directional
        # -- create the 'second half' of the matrix
        dft = data.copy()
        col_product_1 = cols_products[0]
        col_product_2 = cols_products[1]
        dft.rename(columns={f'{col_product_1}':f'{col_product_1}_t'}, inplace=True)
        dft.rename(columns={f'{col_product_2}':f'{col_product_1}'}, inplace=True)
        dft.rename(columns={f'{col_product_1}_t':f'{col_product_2}'}, inplace=True)
        # -- add to dataframe
        data = pd.concat([data,dft],axis=0,sort=False).reset_index(drop=True)
    # -- calculate the Cpp properties now
    data['Cp'] = data.groupby(col_product_1)['Cpp'].transform(sum)
    data['Cpprime'] = data.groupby(col_product_2)['Cpp'].transform(sum)
    data['C'] = data['Cpp'].sum()
    data['Crtp'] = data['Cpp']/((data['Cp']*data['Cpprime'])/data['C'])

    return data

# Calculate crtp, whilst excluding Unspecified products
dft = df_cpp[df_cpp['product_1']!='Unspecified']
dft = dft[dft['product_2']!='Unspecified']
df_crtp = calc_crtp(dft,cols_products=['product_1','product_2'],col_cpp=['Cpp'])

# Show products that relatively co-occur often
df_crtp.sort_values(by=['Crtp'],ascending=False,inplace=True)
"""
q999 = df_crtp['Crtp'].quantile(.999)
df_crtp[df_crtp['Crtp']>q999].sample(n=20)
"""
df_crtp[0:10][['product_1','product_2','Crtp']]
#+END_SRC

For product combinations that are overrepresented against the random benchmark $C_{p} C_{p'}/C$,  $1<\tilde{\pi}_{pp'}<\infty$, whereas for product combinations that are underrepresented against their random benchmark $0<\tilde{\pi}_{pp'}<1$. As a consequence, $\tilde{\pi}_{pp'}$ is distributed with a heavy right-skew. To reduce this skew, we use the following transformation that maps $\tilde{\pi}_{pp'}$ symmetrically around $0.5$ on the interval $[0,1)$:

\begin{equation} \label{e_trans_prox}
\pi_{pp'} = \frac{\tilde{\pi}_{pp'}}{\tilde{\pi}_{pp'}+1}
\tag{6}
\end{equation}

* Patents: RCAs and technology proximities


We can apply this to patent data as well.

At the Growth Lab we have access to the Patstat database, patents from Google Bigquery, patents from PatentView, HistPat, and patents obtained from USPTO publications from 1790 onwards through optical character recognition (all available on the RC Cannon cluster).

** Technological diversification of countries

First we look at diversification of countries, using patents extracted from the Patstat database.

Patstat includes all patents from ~ 1903 onwards.

Below is an outline of what is available in Patstat:

[[https://www.dropbox.com/s/zqgv7fi61c2ip2f/patstat.png?dl=1]]

Below we use an aggregated file created from the Patstat database, containing:

- Year
- Country
- Technology class
- Count (N of patents)

which I put on Dropbox temporarily so we can load it in directly into Google CoLab.

*** Load patent data

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" file :async yes

# Load STATA file into pandas, directly from URL
print('loading patent data')
dfp = pd.read_stata('https://www.dropbox.com/s/nwox3dznoupzm0q/patstat_year_country_tech_inventor_locations.dta?dl=1')
print('patent data loaded')
#+END_SRC


**** Sample of data

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
dfp.sample(n=10)
#+END_SRC


**** What are the first and last years in the data?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
dfp['year'].min()
dfp['year'].max()
#+END_SRC

**** How many countries and technology classes are in the data?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
print('Unique N of counties:')

dfp['country_name'].nunique()

print('Unique N of technologies:')

dfp['tech'].nunique()
#+END_SRC

*** RCAs

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" file :async yes
dfp_rca = calc_rca(data=dfp,
                   country_col='country_name',
                   product_col='tech',
                   time_col='year',
                   value_col='count')
print('patent rcas ready')
#+END_SRC

**** What were Japan and Germany specialized in, in 1960 and 2010?
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# What were Japan and Germany specialized in, in 1960 and 2010?
for country in ['Japan','Germany']:
    for year in [1960, 2010]:
        print(f"---------")
        print(f"\n {country} in {year} \n")
        dft = dfp_rca[dfp_rca['country_name']==country].copy()
        dft = dft[dft['year']==year]
        # --
        dft.sort_values(by=['RCAcpt'],ascending=False)[0:10]
#+END_SRC

*** Technology proximities

What technology classes are most proximate (in 2010)?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# Define Mcp
dfp_rca['Mcp'] = 0
dfp_rca.loc[dfp_rca['RCAcpt']>1,'Mcp'] = 1

# Keep only years 2010
#dft = dfp_rca[ (dfp_rca['year']>=1990) & (dfp_rca['year']<=2020)].copy()
dft = dfp_rca[ (dfp_rca['year']==2010) ].copy()

# Keep only country-product combinations where Mcp == 1 (thus RCAcp > 1)
dft = dft[dft['Mcp']==1]

# Calculate cppt
dfp_cppt = calc_cppt(dft,country_col='country_name',product_col='tech')
print('cppt patent co-occurences and proximities dataframe ready')

# Show most proximate technologies
dfp_cppt.sort_values(by=['phi'],ascending=False)[0:10]
#+END_SRC

(You can use density regressions as well here to predict technological diversification of countries.)

** Technological diversification of cities in the USA

We can also investigate technological diversification at the sub-national level.

Below we're using patent counts per city per technology from 1975 onwards (obtained from patents extracted from the PatentView database). Patents' technologies are defined according to the Cooperative Patent Classification (CPC).

*** Load patent data

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
print('loading patent data')
dfp = pd.read.csv('https://www.dropbox.com/s/th4zqkmuofmg4u3/patentview_class_2022.csv?dl=1')
print('patent data loaded')
#+END_SRC

**** Sample of data

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
dfp.sample(n=10)
#+END_SRC

**** What are the first and last years in the data?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
dfp['year'].min()
dfp['year'].max()
#+END_SRC

**** How many cities (regions) and technology classes are in the data?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
dfp['region'].nunique()
dfp['tech'].nunique()
#+END_SRC

*** RCAs

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# keep 1980 and 2017
dfp_rca = dfp[ (dfp['year']==1980]) | (dfp['year']==2017 )]


dfp_rca = calc_rca(data=dfp_rca,
                   country_col='region',
                   product_col='tech',
                   time_col='year',
                   value_col='count')

# -- calculate
print('patent rcas ready')
#+END_SRC

**** What were Silicon Valley (Santa Clara county) and Detroit (MI - Wayne county) specialized in, in 1980 and 2017?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
for region in ['CA > Santa Clara','MI > Wayne]:
    for year in [1980, 2017]:
        print(f"---------")
        print(f"\n {region} in {year} \n")
        dft = dfp_rca[dfp_rca['region]==region].copy()
        dft = dft[dft['year']==year]
        dft = dft[dft['count']>5]
        # --
        dft.sort_values(by=['RCAcpt'],ascending=False)[0:10]
#+END_SRC


*** Technology proximities (cpc)

What technology classes (cpc classification) are most proximate (in 2010)?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
# Define Mcp
dfp_rca['Mcp'] = 0
dfp_rca.loc[dfp_rca['RCAcpt']>1,'Mcp'] = 1

# Keep only years 2010
#dft = dfp_rca[ (dfp_rca['year']>=1990) & (dfp_rca['year']<=2020)].copy()
dft = dfp_rca[ (dfp_rca['year']==2017) ].copy()

# Keep only country-product combinations where Mcp == 1 (thus RCAcp > 1)
dft = dft[dft['Mcp']==1]

# Calculate cppt
dfp_cppt = calc_cppt(dft,country_col='region',product_col='tech')
print('cppt patent co-occurences and proximities dataframe ready')

# Show most proximate technologies
dfp_cppt.sort_values(by=['phi'],ascending=False)[0:10]
#+END_SRC


* Product space

** Overview
We now have a measure of similarity between products (and patents), which is the core of the product space.

https://atlas.cid.harvard.edu/explore/network?country=114&year=2018&productClass=HS&product=undefined&startYear=undefined&target=Product&partner=undefined

[[~/Dropbox/proj/org_zhtml_projects/product-space-eci-workshop/imgs/product_space_atlas_website.png]]

[[https://www.dropbox.com/s/izag1xf28yldanf/product_space_atlas_website.png?dl=1]]

Below we will explore the product space using Python. You can then directly manipulate the product space and visualize selectively if not possible in the Atlas interface (e.g. only products exported to certain countries).

The Github repo for this is available at [[https://github.com/matteha/py-productspace]].

What we need is information on:

- Edges (ties) between nodes

    Ties between nodes represent the product proximity calculated above. Each product pair has a proximity value, but visualizing all ties, however, would result in a major "hairball".

    To determine which of the ties to visualize in the product space, a 'maximum spanning tree algorithm' is used (to make sure all nodes are connected directly or indirectly) in conjunction with a certain proximity threshold (0.55 minimum conditional probability). The details can be found in the Supplementary Material of Hidalgo et al. (2007) at [[https://science.sciencemag.org/content/suppl/2007/07/26/317.5837.482.DC1]].

    The data on the ties of nodes is available in the Atlas data repository at:
    https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/FCDZBN/QSEETD&version=1.1

    We can directly load it into Python using the link below (temporarily for this session, when using Harvard's dataverse you'd need to sign a short User Agreement form so you can't load data directly from a URL):

    https://www.dropbox.com/s/r601tjoulq1denf/network_hs92_4digit.json?dl=1

- Position of nodes

  + Each node is a product

  + To position them in the product space, Hidalgo et al. (2007) used a spring embedding algorithm (which positions the nodes in such a way that there are as few crossing ties as possible, using physical simulations with force-directed algorithms), followed by hand-crafting the outcome to further visually separate distinct 'clusters' of products.

    The data on the position of nodes (x, y coordinates) is in the same file as the one above with the data on ties (network_hs92_4digit.json).

    We will use this fixed layout for now (James and Yang will deal with different ways to visualize multi-dimensional data in 2D/3D, e.g. with machine learning, UMAP).

- Size of nodes

    The size in the product space represents the total $ in world trade, but one can also use other attributes of nodes (e.g. if nodes are industries, the size could be total employment).
  
- Color of nodes

    In the product space the node color represents major product groups (e.g. Agriculture, Chemicals) following the Leamer classification. The node coloring data is available in the Atlas data repository at:
    https://dataverse.harvard.edu/dataverse/atlas?q=&types=files&sort=dateSort&order=desc&page=1

    We can directly load it into Python using the link below (again, temporary for this session):
    https://www.dropbox.com/s/rlm8hu4pq0nkg63/hs4_hex_colors_intl_atlas.csv?dl=1

** Product space in Python

*** Function to create product space
The function below creates the product space. It uses the 'networkx' package.
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

def create_product_space(df_plot_dataframe=None,
                         df_plot_node_col=None,
                         df_node_size_col=None):

    # No legend, not properly coded yet
    show_legend = 0

    # Copy dataframe so original won't be overwritten
    df_plot =  df_plot_dataframe.copy()

    NORMALIZE_NODE_SIZE = 1
    if NORMALIZE_NODE_SIZE == 1:

        """
        The distribution of export values is highly skewed, which makes it hard to visualize
        them properly (certain products will overshadow the rest of the network).


        We create a new size column below in which we normalize the export values.
        """

        ### Normalize node size (0.1 to 1)

        def normalize_col(dft,col,minsize=0.1,maxsize=1):
            """
            Normalizes column values with largest and smallest values capped at min at max
            For use in networkx

            returns pandas column
            """

            alpha = maxsize-minsize
            Xl = dft[dft[col]>0][col].quantile(0.10)
            Xh = dft[dft[col]>0][col].quantile(0.95)
            dft['node_size'] = 0
            dft.loc[ dft[col]>=Xh,'node_size'] = maxsize
            dft.loc[ (dft[col]<=Xl) & (dft[col]!=0),'node_size'] = minsize
            dft.loc[ (dft[col]<Xh) & (dft[col]>Xl),'node_size'] = ((alpha*(dft[col]-Xl))/(Xh-Xl))+(1-alpha)
            dft.loc[ (dft[col]<Xh) & (dft[col]>0),'node_size'] = ((alpha*(dft[col]-Xl))/(Xh-Xl))+(1-alpha)

            return dft['node_size']

        df_plot['node_size'] = normalize_col(df_plot,df_node_size_col,minsize=0.1,maxsize=1)

    ADD_COLORS_ATLAS = 1
    if ADD_COLORS_ATLAS == 1:

        # First add product codes from original file (full strings were used for illustrative purposes above but we need the actual codes to merge data from other sources, e.g. node colors)
        df_plot = pd.merge(df_plot,df_orig[['product_name','product_code']].drop_duplicates(),how='left',on='product_name')
        dft = pd.read_csv('https://www.dropbox.com/s/rlm8hu4pq0nkg63/hs4_hex_colors_intl_atlas.csv?dl=1')

        # Transform product_code into int (accounts for missing in pandas, if necessary)
        # keep only numeric product_codes (this drops 'unspecified' as well as services for now;
        # - as the latter needs a separate color classification)
        df_plot = df_plot[df_plot['product_code'].astype(str).str.isnumeric()]
        # -- also drop 9999 product code; unknown
        df_plot = df_plot[df_plot['product_code'].astype(str)!='9999']
        # -- to allow merge, rename and transform both variables into int
        dft['hs4'] = dft['hs4'].astype(int)
        df_plot['product_code'] = df_plot['product_code'].astype(int)
        if 'color' in df_plot.columns:
            df_plot.drop(['color'],axis=1,inplace=True,errors='ignore')
        df_plot = pd.merge(df_plot,dft[['hs4','color']],how='left',left_on='product_code',right_on='hs4')
        # drop column merged from dft
        df_plot.drop(['hs4'],axis=1,inplace=True,errors='ignore')

        # CREATE_LEGEND = 1
        # if CREATE_LEGEND == 1:
        #     # Atlas classification products
        #     df_temp = pd.read_csv('https://raw.githubusercontent.com/cid-harvard/classifications/master/product/HS/IntlAtlas/out/hs92_atlas.csv')
        #     df_temp.rename(columns={'Unnamed: 0': 'internal_code'}, inplace=True)
        #     # -- keep sections
        #     df_temp1 = df_temp[df_temp['level']=='section'].copy()
        #     # -- keep 4 digit
        #     df_temp2 = df_temp[df_temp['level']=='2digit'].copy()
        #     # -- keep 4 digit
        #     df_temp3 = df_temp[df_temp['level']=='4digit'].copy()
        #     # -- merge parent id of parent id of 4 digits (= 2digit)
        #     # ---- remake to float
        #     df_temp3['parent_id'] = df_temp3['parent_id'].astype(object)
        #     df_temp2['internal_code'] = df_temp2['internal_code'].astype(object)
        #     df_temp3t = pd.merge(df_temp3,df_temp2[['internal_code','parent_id']],how='left',left_on='parent_id',right_on='internal_code',indicator=True)
        #     # -- now merge parent_id_y to internal code of df_temp1
        #     df_temp3t.drop(['_merge'],axis=1,inplace=True)
        #     df_temp3t2 = pd.merge(df_temp3t,df_temp1[['internal_code','name']],how='left',left_on='parent_id_y',right_on='internal_code',indicator=True)
        #     # keep only relevant columns
        #     df_temp4 = df_temp3t2[['code','name_x','name_y']]
        #     df_temp5 = df_temp4[['code','name_y']]
        #     df_temp5.rename(columns={'code': 'product'}, inplace=True)
        #     df_temp5.rename(columns={'name_y': 'name_sector_atlas'}, inplace=True)
        #     # drop XXXX / services (not in product space)
        #     drop_categories = ['XXXX','unspecified','travel','transport','ict','financial']
        #     df_temp5 = df_temp5[ ~(df_temp5['product'].isin(drop_categories))]

        #     # add to df_temp_plot
        #     df_temp_plott = df_plot.copy()
        #     df_temp_plott['product_code'] = df_temp_plott['product_code'].astype(float)
        #     df_temp5['product'] = df_temp5['product'].astype(float)
        #     df_temp_plot3 = pd.merge(df_temp_plott,df_temp5,how='left',left_on='product_code',right_on='product')
        #     df_temp_plot3.drop_duplicates(subset='color',inplace=True)
        #     df_temp_plot3 = df_temp_plot3[['color','name_sector_atlas']]

        #     # create color dict for legend
        #     color_dict = {}
        #     df_temp_plot3.reset_index(inplace=True,drop=True)
        #     for ind, row in df_temp_plot3.iterrows():
        #         color_dict[row['name_sector_atlas']] = row['color']

        #     """
        #     def build_legend(data):
        #         # Build a legend for matplotlib plt from dict
        #         legend_elements = []
        #         for key in data:
        #             legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
        #                                             markerfacecolor=data[key], markersize=10))
        #         return legend_elements
        #     fig,ax = plt.subplots(1)
        #     #ax.add_patch(rect) # Add the patch to the Axes
        #     legend_elements = build_legend(color_dict)
        #     ax.legend(handles=legend_elements, loc='upper left')
        #     plt.show()
        #     """

    ADD_NODE_POSITIONS_ATLAS = 1
    if ADD_NODE_POSITIONS_ATLAS == 1:

        # Load position of nodes (x, y coordinates of nodes from original Atlas file)
        import urllib.request, json
        with urllib.request.urlopen("https://www.dropbox.com/s/r601tjoulq1denf/network_hs92_4digit.json?dl=1") as url:
            networkjs = json.loads(url.read().decode())

    CREATE_NETWORKX_OBJECT_AND_PLOT = 1
    if CREATE_NETWORKX_OBJECT_AND_PLOT == 1:

        # Convert json into python list and dictionary
        nodes = []
        nodes_pos = {}
        for x in networkjs['nodes']:
            nodes.append(int(x['id']))
            nodes_pos[int(x['id'])] = (int(x['x']),-int(x['y']/1.5))

        # Define product space edge list (based on strength from the json)
        edges = []
        for x in networkjs['edges']:
            if x['strength'] > 1 or 1 == 1:
                edges.append((int(x['source']),int(x['target'])))
        dfe = pd.DataFrame(edges)
        dfe.rename(columns={0: 'src'}, inplace=True)
        dfe.rename(columns={1: 'trg'}, inplace=True)

        # Only select edges of nodes that are also present in product space
        dfe2 = pd.DataFrame(np.append(dfe['src'].values,dfe['trg'].values)) # (some products may not be in there)
        dfe2.drop_duplicates(inplace=True)
        dfe2.rename(columns={0: 'node'}, inplace=True)
        dfn2 = pd.merge(df_plot,dfe2,how='left',left_on=df_plot_node_col,right_on='node',indicator=True)

        # Drop products from this dataframe that are not in product space
        dfn2 = dfn2[dfn2['_merge']=='both']

        # Create networkx objects in Python

        # G object = products that will be plotted
        G=nx.from_pandas_edgelist(dfn2,'product_code','product_code')

        # G2 object = all nodes and edges from the original product space
        # - Those that are not plotted will be gray in the background,
        # - e.g. products for which there is no info
        G2=nx.from_pandas_edgelist(dfe,'src','trg')

        # Add node attributes to networkx objects
        # - Create a 'present' variable which indicates that these products are present in product space,
        # - as not all products in product space are present in the data to be plotted
        # - (e.g. because we could filter only to plot products with more than >$40 million in trade)
        df_plot['present'] = 1
        ATTRIBUTES = ['node_size'] + ['color'] + ['present']
        for ATTRIBUTE in ATTRIBUTES:
            dft = df_plot[[df_plot_node_col,ATTRIBUTE]]
            dft['count'] = 1
            dft = dft.groupby([df_plot_node_col,ATTRIBUTE],as_index=False)['count'].sum()
            #** drop if missing , and drop duplicates
            dft.dropna(inplace=True)
            dft.drop(['count'],axis=1,inplace=True)
            dft.drop_duplicates(subset=[df_plot_node_col,ATTRIBUTE],inplace=True)
            dft.set_index(df_plot_node_col,inplace=True)
            dft_dict = dft[ATTRIBUTE].to_dict()
            for i in sorted(G.nodes()):
                try:
                    #G.node[i][ATTRIBUTE] = dft_dict[i]
                    G.nodes[i][ATTRIBUTE] = dft_dict[i]
                except Exception:
                    #G.node[i][ATTRIBUTE] = 'Missing'
                    G.nodes[i][ATTRIBUTE] = 'Missing'
            for i in sorted(G2.nodes()):
                try:
                    #G2.node[i][ATTRIBUTE] = dft_dict[i]
                    G2.nodes[i][ATTRIBUTE] = dft_dict[i]
                except Exception:
                    #G2.node[i][ATTRIBUTE] = 'Missing'
                    G2.nodes[i][ATTRIBUTE] = 'Missing'

        # Cross-check that attributes have been added correctly
        # nx.get_node_attributes(G2,df_color)
        # nx.get_node_attributes(G,df_color)

        # Create color + size lists which networkx uses for plotting
        groups = set(nx.get_node_attributes(G2,'color').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = G.nodes()
        nodes2 = G2.nodes()
        #colorsl = [G.node[n]['color'] for n in nodes]
        colorsl = [G.nodes[n]['color'] for n in nodes]
        #colorsl2 = [G2.node[n]['color'] for n in nodes2]
        colorsl2 = [G2.nodes[n]['color'] for n in nodes2]
        SIZE_VARIABLE = 'node_size'
        #sizesl = [G.node[n][SIZE_VARIABLE] for n in nodes]
        sizesl = [G.nodes[n][SIZE_VARIABLE] for n in nodes]
        # Adjust value below to increase the PLOTTED size of nodes, depending on the resolution of the final plot
        # (e.g. if you want to zoom in into the product space and thus set a higher resolution, you may want to set this higher)
        #sizesl2 = [G.node[n]['node_size']*350 for n in nodes]
        sizesl2 = [G.nodes[n]['node_size']*350 for n in nodes]

        # Create matplotlib object to draw the product space
        f = plt.figure(1,figsize=(20,20))
        ax = f.add_subplot(1,1,1)

        # turn axes off
        plt.axis('off')

        # set white background
        f.set_facecolor('white')

        # draw full product space in background, transparent with small node_size
        nx.draw_networkx(G2,nodes_pos, node_color='gray',ax=ax,node_size=10,with_labels=False,alpha=0.1)

        # draw the data fed in to the product space
        nx.draw_networkx(G,nodes_pos, node_color=colorsl,ax=ax,node_size=sizesl2,with_labels=False,alpha=1)

        # save file
        # plt.savefig(output_dir_image)

        # show the plot
        plt.show()

        # if show_legend == 1:
        #     # show legend as well
        #     def build_legend(data):
        #         # Build a legend for matplotlib plt from dict
        #         legend_elements = []
        #         for key in data:
        #             legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
        #                                             markerfacecolor=data[key], markersize=10))
        #         return legend_elements
        #     fig,ax = plt.subplots(1)
        #     #ax.add_patch(rect) # Add the patch to the Axes
        #     legend_elements = build_legend(color_dict)
        #     ax.legend(handles=legend_elements, loc='upper left')
        #     plt.show()

print('defined product space function, ready to plot')

#+END_SRC

*** Run function from github url / local file directly

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

import requests
url = 'https://raw.githubusercontent.com/cid-harvard/py-productspace/master/create_product_space_v2.py'
r = requests.get(url)
#print(r.content)
exec(r.content)
print('product space code imported')
execfile('/Users/admin/Dropbox/proj/git_clones/py-productspace/create_product_space_v2.py')
print("RAN LOCAL FILE")
#+END_SRC


*** Visualizing data in the product space

First we select the country we which to visualize. We'll search for Saudi Arabia below, using the 'audi' string to find out the spelling of the country in the dataset, and we input that country name when defining the dataframe of the product space ('df_ps').

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# Find out what 'country_name corresponds to Saudi Arabia
STRING = 'audi'
df_rca[df_rca['country_name'].str.contains(STRING)][['country_name']].drop_duplicates()
# result: Saudi Arabia'

#+END_SRC


**** Country, RCA, year, export value selections
Next we define what trade properties of Saudi Arabia we want to visualize. The example below visualizes specialiation in 2005 (year=2005, RCAcpt>1) of only those products with at least 40 million in trade value.

This data preparation happens outside of the product space function so you can inspect the dataframe before plotting.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes

# Select country
COUNTRY_STRING = 'Saudi Arabia'
df_ps = df_rca[df_rca['country_name']==COUNTRY_STRING].copy()

# Cross-check
if df_ps.shape[0] == 0:
    print('Country string set above does not exist in data, typed correctly?')
    STOP

# Select year
df_ps = df_ps[df_ps['year']==2005].copy()

# Select RCA > 1
df_ps = df_ps[df_ps['RCAcpt']>1]

# Keep only relevant columns
df_ps = df_ps[['product_name','product_code','export_value']]

# Keep only products with minimum value threshold
exports_min_threshold = 40000000
df_ps = df_ps[df_ps['export_value']>exports_min_threshold]

# Show resulting dataframe
df_ps.sample(n=5)

# Save file
df_ps.to_csv(f'~/Dropbox/df_product_space.csv',index=False, encoding='utf-8')
print('saved')

print('ready to plot in product space')

#+END_SRC

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

# Plot in the product space
create_product_space(df_plot_dataframe=df_ps,
                     df_plot_node_col='product_code',
                     df_node_size_col='export_value',
                     )


#+END_SRC

**** Plot from file saved (to use in R) ::noexport::
#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
execfile('/Users/admin/Dropbox/proj/git_clones/py-productspace/create_product_space_v2.py')
print("RAN LOCAL FILE")

create_product_space(df_plot_dataframe_filename='~/Dropbox/df_product_space.csv',
                     df_plot_dataframe=df_ps,
                     df_plot_node_col='product_code',
                     df_node_size_col='export_value',
                     output_image_file ='/Users/admin/Dropbox/testnetwork.png'
                     )
#+END_SRC

#+RESULTS:

**** Product space with legend

Below is a legend of the product space. There's also a 'show legend' option in the 'create product space' function but this option needs to be updated.

[[https://www.dropbox.com/s/lf4lf8ktqahnovg/Selection_032.png?dl=1]]

To see exactly what node represents what product, use the Atlas for now by hovering with the mouse over a node:

https://atlas.cid.harvard.edu/explore/network?country=186&year=2018&productClass=HS&product=undefined&startYear=undefined&target=Product&partner=undefined

(This can also be implemented through Python by exporting to html instead of as an image, but not implemented above yet)

* ---------------- Break: Excercise 1 ------------------


** What product does Ukraine export most in 1995? (excluding services such as 'transport', 'ict' etc)

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
#+END_SRC

** What products is Ukraine specialized in in 1995 and 2005 and how much do they export of these?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
#+END_SRC

** Which product is most related to the product 'Stainless steel wire'?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
#+END_SRC

** Plot Ukraine in the product space in 1995.

How would you characterize Ukraine's position in the product space?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
#+END_SRC

** Plot Ukraine in the product space in 2015.

Do you notice a difference with 1995?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes



#+END_SRC

** Plot your own country across different years in the product space. Do the results make sense? Do you notice any patterns?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes



#+END_SRC

* Predicting diversification of countries: densities / density regressions

Shreyas will cover this.

** Subheader :noexport:

#+BEGIN_COMMENT
Dont add :noexport to main heading, then cant export to ipynb, 'killed buffer' error.
#+END_COMMENT
    Does a country's position in the the product space predict what products it diversifies into in the future? Indeed it does, according to Hidalgo et al. (2007) and many other studies that have followed. If a product is in close proximity to your current (export) basket of products, you are more likely to diversify into that product as a result: monkeys can only jump to the nearest branch in a tree.

    (You can also see this for yourself using the code above, by plotting the same country in the product space across subsequent years. Best done using the SITC trade classification that goes back further in time into the 1970s, rather than the HS classification used above which starts only in 1995).

    To test this empirically, one can perform so called 'density regressions'.

    For every possible country-product combination, you calculate the extent to which one's existing product portfolio is proximate (using the product proximities calculated earlier) to it, which is refered to as 'density'.

    You then test whether density predicts whether country-product combinations that were not present in $t$ are actually present in $t + 1$.

    We will do so below.

    ** Prepare product-product-proximity matrix, all years
    First we create a matrix of all possible product combinations in all years and we add proximities to it (we take the earlier product proximity matrix that we created, which was done using 1995 data, and make sure it also contains proximities to products that were not present at all in 1995. To avoid calculation problems with missing values later).

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

    # Create product * product matrix and add proximity for each product
    # -- in long format
    products = df_rca['product_name'].unique()
    combs = list(combinations(products,2))
    df_pp = pd.DataFrame(combs,columns=['product_1','product_2'])
    # df_pp.shape # should be N products * N products

    # make it asymmetrical
    dft = df_pp.copy()
    dft.rename(columns={f'product_2':f'product_1t'}, inplace=True)
    dft.rename(columns={f'product_1':f'product_2'}, inplace=True)
    dft.rename(columns={f'product_1t':f'product_1'}, inplace=True)
    df_pp = pd.concat([df_pp,dft],axis=0)

    # add proximities
    df_pp = pd.merge(df_pp,df_cppt[['product_1','product_2','phi']],how='left',on=[f'product_1','product_2'])

    # set proximity to 0 if missing (preferably all products are in matrix)
    df_pp.loc[df_pp['phi'].isnull(),'phi'] = 0

    print('product-product proximity matrix for all years ready')
    #+END_SRC

    ** Old code :noexport:

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
    use_crtp = 0
    if use_crtp == 1:
        df_crtp[0:10][['product_1','product_2','Crtp']]
        df_pp = pd.merge(df_pp,df_crtp[['product_1','product_2','Crtp']],how='left',on=[f'product_1','product_2'])
        # set proximity to 0 if missing (preferably all products are in matrix)
        # ; depends on how proximity matrix was created: (based on rca matrix for which year)
        df_pp.loc[df_pp['Crtp'].isnull(),'Crtp'] = 0
        ! need to standardize these

    use_shreyas_proximity = 0
    if use_shreyas_proximity == 1:
        prox_df[0:20]
        prox_df.shape
        prox_df[0:20]
        df_pp.shape
        df_pp = prox_df.copy()
    #+END_SRC


    ** Calculate density
    Next, for each country, we take the portfolio of underdeveloped or ('not present') products in $t$ (1996 in the example below).

    Following Hidalgo et al (2007) we define:

    - 'underdeveloped' as those country-product combinations with with RCA < 0.5.
    - 'developed' as those country-product combinations with an RCA > 1.

    (Those with RCAs between 0.5 and 1 Hidalgo et al refer to as 'inconclusive')

    We then use this information in conjunction with the product proximity matrix above to calculate density following Hidalgo et al:

    \begin{equation} \label{density}
    \omega_{cj} = \frac{
    \sum \limits_{i} \chi_{i} \phi_{ij}
    }
    {\sum \limits_{i} \phi_ij }
    \tag{7}
    \end{equation}

    where $\omega_{ci}$ is the density around product $j$ for the $c^_th$ country, $\chi_{i}$ = 1 if RCA > 1 and 0 otherwise, and $\phi_{ij}$ is the matrix of conditional proximities between products that we created earlier.

    A density of 0.44 would imply that 44% of the neighbouring space of the product seems to be developed in the country.

    For each product-country combination that is 'underdeveloped', we will create a density value below.

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

    def calc_density_hidalgo_et_al(rca_dataframe=None,
                    region_col = None,
                    product_col = None,
                    rca_col = None,
                    year = None,
                    underdeveloped_maximum_rca_threshold = 0.5,
                    developed_minimum_rca_threshold = 1):

        """
        Calculaties densities for product-country combinations.

        Returns a pandas dataframe.

        """

        # Keep only country-product information in year specified
        df_d = rca_dataframe[rca_dataframe['year']==year].copy()

        # drop if countries have 0 exports in whole year
        # - don't calculate densities for these: error in data
        df_d['exports_sum'] = df_d.groupby([region_col])['export_value'].transform('sum')
        df_d = df_d[df_d['exports_sum']!=0]

        # Keep only necessary columns
        df_d = df_d[[region_col,product_col,rca_col]]

        # This will be the country-product density dataframe to which densities are appended below
        df_cpd = pd.DataFrame()

        # We will loop over regions (countries) to save memory
        REGIONS = df_d[region_col].unique()
        indexL = 10
        for index,REGION in enumerate(REGIONS):
            if index == indexL:
                print(f'Done region {index} out of {len(REGIONS)}')
                indexL = indexL + 10
            df_dc = df_d[df_d[region_col]==REGION].copy()

            # For the sake of completion: we want to add all products to the matrix: products that
            # have not (yet) been present in a country are now not in the rca matrix
            # We thus use the pp-matrix for this
            products_exclude_from_not_developed = df_dc[df_dc[rca_col]>=underdeveloped_maximum_rca_threshold][product_col].unique()
            products_not_developed = [x for x in df_pp['product_1'].unique() if x not in products_exclude_from_not_developed]

            # Merge this into proximity matrix (in long form)
            # -- keep only proximities for those products in the country's 'underdeveloped' portfolio
            df_ppt = df_pp[df_pp['product_1'].isin(products_not_developed)].copy()

            # now add products the country does have developed (RCA > 1 in t)
            df_developed = df_dc[df_dc['RCAcpt']>developed_minimum_rca_threshold]
            dft = pd.merge(df_ppt,df_developed,how='left',left_on='product_2',right_on=product_col)
            dft.rename(columns={f'RCAcpt':f'RCAcpt_product2'}, inplace=True)

            # Density includes only those products 'developed/present' products (RCA > 1)
            dft['phi_include'] = 0
            dft.loc[ (dft['RCAcpt_product2']>1),'phi_include'] = dft['phi']

            # Take the sum of these for product 1
            dft['density_sum'] = dft.groupby(['product_1'])['phi_include'].transform('sum')

            # Divide this density_sum by sum of all densities as in Hidalgo et al
            dft['density_sum_all'] = dft.groupby(['product_1'])['phi'].transform('sum')
            dft['density'] = dft['density_sum'] / dft['density_sum_all']
            dft.loc[dft['density'].isnull(),'density'] = 0 # 0 if missing, no sum

            # Now drop information on product 2: keep only one observation per product 1
            dft.drop_duplicates(subset='product_1',inplace=True)

            # add region information again
            dft[region_col] = REGION

            # Keep only relevant columns
            dft = dft[[region_col,'product_1','density']]

            # add to country-product density dataframe
            df_cpd = pd.concat([df_cpd,dft],axis=0)

        print(f'country-product densities finished for year {year}')

        return df_cpd

    # -- Create country-product densities dataframe
    # Loop over countries now, takes about 2 minutes
    df_cpd = calc_density_hidalgo_et_al(rca_dataframe=df_rca,
                    region_col = 'country_name',
                    product_col = 'product_name',
                    rca_col = 'RCAcpt',
                    year = 1996,
                    underdeveloped_maximum_rca_threshold = 0.5,
                    developed_minimum_rca_threshold = 1)

    #+END_SRC

    ** Add information on product diversification of countries in t + 1
    For each country we now have a vector of underdeveloped products and their corresponding density. Next we add information on whether countries actually developed those products in $t + 1$ (i.e. if they have an RCA > 1 in $t + 1$).

    We will use 2005 as $t + 1$ below.

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

    # Now add information on whether these products are present 10 years later
    # -- again using the rca matrix
    df_future = df_rca[df_rca['year']==2005].copy()

    # keep only relevant columns
    df_future = df_future[['country_name','product_name','RCAcpt','export_value']]

    # tag and drop countries with no exports at all: error in data
    df_future['exports_sum'] = df_future.groupby('country_name')['export_value'].transform('sum')
    df_cpdf = pd.merge(df_cpd,df_future,how='left',left_on=[f'country_name',f'product_1'],right_on=['country_name','product_name'],indicator=True)
    df_cpdf = df_cpdf[df_cpdf['exports_sum']!=0] # (none in 2005 when dropped in 1996 density calculations)

    # Remove those with RCA between 0.5 and 1: 'inconclusive' in Hidalgo et al 2007
    df_cpdf = df_cpdf[ (df_cpdf['RCAcpt']<0.5) | (df_cpdf['RCAcpt']>1) ]

    # Tag if product was developed or not
    df_cpdf['present'] = 0
    df_cpdf.loc[df_cpdf['RCAcpt']>1,'present'] = 1
    #+END_SRC


    *** Cross-check density / diversification in t + 1 information Shreyas :noexport:

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

    df_s = df_rca[df_rca['year']==1996].copy()
    # keep only relevant columnsz
    """
    df_s = df_s[['year','country_name','product_name','RCAcpt','export_value']]
    df_s[0:10]
    """
    df_s = df_s[['year','country_name','product_name','export_value']]

    from ecomplexity import ecomplexity
    from ecomplexity import proximity

    # To use py-ecomplexity, specify the following columns
    trade_cols = {'time':'year',
                'loc':'country_name',
                'prod':'product_name',
                'val':'export_value'}

    # Then run the command
    df_ec = ecomplexity(df_s, trade_cols)
    df_ec[0:20]

    # Keep only 'undeveloped' in t
    df_ec[ (df_ec['country_name']=='Luxembourg') & (df_ec['product_name']=='ICT')]
    df_ec[ (df_ec['country_name']=='Luxembourg') ]['rca'].value_counts(normalize=False)
    df_ec[ (df_ec['country_name']=='Luxembourg') ]
    """
    MISSING FOR LUXEMBOURG? WHY?
    NO EXPORTS AT ALL
    SET RCA TO ZERO INSTEAD?
    OR NOT, SINCE THEY HAVE NO EXPORTS AT ALL SO MISTAKE!d
    INSTEAD OF DROPPING
    THAT EXPLAINS DIF DENSITIES
    """
    df_ec[df_ec['export_value']==0].sample(n=20)

    df_ec = df_ec[df_ec['rca']<0.5]
    df_ec.shape


    df_ec[ (df_ec['country_name']=='Luxembourg') & (df_ec['product_name']=='ICT')]
    df_cpd.shape

    # Now add information on whether these products are present 10 years later
    # -- again using the rca matrix
    #df_d = df_rca[df_rca['country_name']==COUNTRY].copy()
    df_future = df_rca[df_rca['year']==2005].copy()
    # which countries have 0 exports in years?
    df_ce = df_rca.groupby(['year','country_name'],as_index=False)['export_value'].sum()
    df_ce[df_ce['country_name']=='Luxembourg']
    # keep only relevant columns
    # df_future = df_future[df_future['country_name']==COUNTRY]
    df_future = df_future[['country_name','product_name','RCAcpt']]

    df_cpdf2 = pd.merge(df_ec,df_future,how='left',left_on=['country_name',f'product_name'],right_on=['country_name','product_name'],indicator=True)
    df_cpdf2['_merge'].value_counts()

    # tag if present or not
    # Remove those with RCACPT between 0.5 and 1: 'inconclusive' in Hidalgo et al 2007
    df_cpdf2 = df_cpdf2[ (df_cpdf2['RCAcpt']<0.5) | (df_cpdf2['RCAcpt']>1) ]
    df_cpdf2['present'] = 0
    df_cpdf2.loc[df_cpdf2['RCAcpt']>1,'present'] = 1
    df_cpdf2.shape
    df_cpdf2['present'].value_counts(normalize=False)

    """
    COUNTRY = 'Bangladesh'
    PRODUCT = 'Woven fabrics of cotton of < 85% weighing > 200 g/m2'
    df_cpdf[ (df_cpdf.country_name==COUNTRY) & (df_cpdf.product_name == PRODUCT) ]
    df_cpdf3[ (df_cpdf3.country_name==COUNTRY) & (df_cpdf3.product_name == PRODUCT) ]
    """

    # merge to original
    CROSS_CHECK = 1
    if CROSS_CHECK == 1:

        df_cpdf3 = pd.merge(df_cpdf,df_cpdf2,how='outer',on=['country_name','product_name'])
        df_cpdf3[['density_x','density_y']].sample(n=20)
        df_cpdf3[['present_x','present_y']].sample(n=20)
        df_cpdf3[['density_x','density_y']].corr()
        pd.crosstab(df_cpdf3['density_x'],df_cpdf3['density_y'])

        df_cpdf3['present_x'].value_counts(normalize=False)
        df_cpdf3['present_y'].value_counts(normalize=False)

        SAME NOW (ALMOST) AFTER DROPPING COUNTRIES WITHOUT EXPORTS AT ALL

        # Some 'present' is there in shreyas but not in this code?
        WHY?

        df_cpdf3[(df_cpdf3['present_x']!= df_cpdf3['present_y']) ][0:30]

        df_cpdf3[(df_cpdf3['present_x']==1) & (df_cpdf3['present_y'].isnull()) ][0:30]
        HOW COME PRESENT?

        df_cpdf3[(df_cpdf3['present_x']==1) & (df_cpdf3['present_y']==0) ][0:30]


        df_cpdf3[(df_cpdf3['present_y']==1) & (df_cpdf3['present_x']==0) ][0:30]
        df_cpdf3[(df_cpdf3['present_y']==1) & (df_cpdf3['present_x'].isnull()) ][0:30]

        df_cpdf3[(df_cpdf3['present_y']==1) & (df_cpdf3['present_x'].isnull()) ][0:30]


        print(f'df_cpdf2 shreyas ready')
    #+END_SRC

    ** Plot density distribution
    Below we plot the density distribution in $t + 1$ of not-developed and developed products.

    Density is generally higher for those products that were developed between $t$ and $t + 1$ than for those that were not.

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
    dfa = pd.DataFrame()
    for present in [0,1]:
        dft = df_cpdf[df_cpdf['present']==present].copy()
        dft.shape
        #dfo = pd.DataFrame([0])
        li = []
        for x in range(1,11,1):
            x_min = (x/10)-0.1
            x_max= (x/10)
            if x_max == 1:
                x_max = 1.01
            sh_in_density = dft[ (dft['density']>=x_min) & (dft['density']<x_max)].shape[0]/dft.shape[0]
            li.append([f'{round(x_min,2)}-{round(x_max,2)}', sh_in_density])
        dfo = pd.DataFrame(li)
        dfo.index=dfo[0]
        dfo.drop(0,axis=1,inplace=True)
        dfo.rename(columns={1:f'{present}'}, inplace=True)
        dfa = pd.concat([dfa,dfo],axis=1)

    dfa.rename(columns={'0':f'not_developed'}, inplace=True)
    dfa.rename(columns={'1':f'developed'}, inplace=True)
    dfa.plot.bar()
    #+END_SRC

    ** Density regression
    Finally we run a simple density regression. We will use the 'statsmodels' package in Python for this (imported at the beginning of the notebook in the first code cell).

    Different packages are available for this in Python, linearmodels, statsmodels, panelOLS (but removed from pandas), and so on.

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes

    # Define X and Y columns / arrays from dataframe
    X_cols = ['density']
    Y_col = ['present']
    # sub-select the columns
    X = df_cpdf[X_cols]
    Y = df_cpdf[Y_col]
    # Add constant
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    # Show results
    print(results.summary())
    #+END_SRC

    (To get stronger results, one could also include only the closest-occupied products' proximity, for instance (see Hidalgo et al's 2007 Supplementary Section).)

    ** Model with fixed effects

    *** Including dummies

    Include FE by including dummies for the variable in question.

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

    # Country fixed effects in statsmodels
    countries_d = pd.get_dummies(df_cpdf['country_name'],drop_first=True)
    # add 'd_' in front of variables
    countries_d.columns = ['d_'+col for col in countries_d.columns]
    # add dummies to main dataframe
    df_cpdf2 = pd.concat([df_cpdf,countries_d],axis=1)
    ##
    X_cols = ['density'] + [col for col in df_cpdf2.columns if 'd_' in col]
    Y_col = ['present']
    ##
    X = df_cpdf2[X_cols]
    Y = df_cpdf2[Y_col]
    # Add constant
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    print(f'fitting')
    results = model.fit()
    print(f'ready')
    # Show results
    print(results.summary())

    #+END_SRC

    *** Demeaning

    Much faster: One can also demean instead of including dummies, as done below.

    #+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
    # (better to do this in STATA)
    MEAN = df_cpdf['density'].mean()
    df_cpdf2 = df_cpdf - df_cpdf.groupby(df_cpdf['country_name']).transform('mean') + MEAN
    ##
    X_cols = ['density']
    Y_col = ['present']
    ##
    X = df_cpdf2[X_cols]
    Y = df_cpdf2[Y_col]
    # Add constant
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    print(f'fitting')
    results = model.fit()
    print(f'ready')
    # Show results
    print(results.summary())
    #+END_SRC

    (Adjust standard errors afterwards).


* Calculating Economic Complexity / Product Complexity
We know from the product space and density regressions how products are related to one another and how that matters for diversification of countries.

The next step is to look at which parts of the product space are most interesting to ultimately reach / diversify into. Generally complex products are located in the center of the product space, and countries with a higher economic complexity tend to have higher economic growth.

[[file:imgs/complex_products_in_product_space.png]]

[[https://www.dropbox.com/s/a231jw76yocjkkr/complex_products_in_product_space.png?dl=1]]

Recall from the lecture that the economic complexity index (ECI) and product complexity index (PCI) measures are derived from an iterative method of reflections algorithm on country diversity and product ubiquity (Hidalgo Hausmann 2009), or finding the eigenvalues of a country-product matrix (Mealy et al. 2019)

[[~/Dropbox/proj/org_zhtml_projects/product-space-eci-workshop/imgs/countries_products_eci.png]]

https://www.dropbox.com/s/dte4vwgk4tvj3rd/countries_products_eci.png?dl=1

The STATA package to calculate this - by Sebastian Bustos and Muhammed Yildirim - is available at:

https://github.com/cid-harvard/ecomplexity

The Python package to calculate this - by Shreyas Gadgin Matha - is available at https://github.com/cid-harvard/py-ecomplexity

The R package to calculate this, by Mauricio Vargas, Carlo Bottai, Diego Kozlowski, Nico Pintar, The World Bank, Open Trade Statistics, is available at:

https://cran.r-project.org/web/packages/economiccomplexity/index.html

(When using other software, e.g. Excel without having access to these packages, one can also calculate ECI by directly downloading the PCI value for every product from the Atlas Dataverse repository - the ECI of a country is the mean of the PCI values of the products it has a comparative advantage in).

** Using the 'py-ecomplexity' package

*** Installation
One can install it by pointing pip (package-management system in Python) to the respective library, using the following command:

#+BEGIN_COMMENT
#!pip show networkx
#!pip install networkx==2.2
#+END_COMMENT

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
!pip install ecomplexity
print('installed py-ecomplexity')
#+END_SRC

*** Usage
We will again use again the original trade dataset (df_orig), below.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

from ecomplexity import ecomplexity
from ecomplexity import proximity

# To use py-ecomplexity, specify the following columns
trade_cols = {'time':'year',
              'loc':'country_name',
              'prod':'product_name',
              'val':'export_value'}

# Then run the command
print('calculating ecomplexity')

# only 2000 now for time sake; economplexity will calculate the values for each year

dft  = df_orig[df_orig['year']==2000]

df_ec = ecomplexity(dft, trade_cols)

print('finished calculating')

# # Keep selected columns
# df_ec = df_ec[[[['country_name',
#                'product_name',
#                'product_code',
#                'export_value',
#                'year',
#                'pci',
#                'eci']]]]

# Show results
df_ec.sample(n=10)

#+END_SRC


*** Cross-check with R ecomplexity package ::noexport::

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
"""
dassda
"""
dft = df_ec[df_ec['year']==2000].copy()
dft.drop_duplicates(subset='country_name',inplace=True)
dft.sort_values(by=['eci'],ascending=False,inplace=True)
dft.reset_index(inplace=True,drop=True)
dft[0:10][['country_name','eci']]


dft = df_ec[df_ec['year']==2000].copy()
dft.drop_duplicates(subset='product_name',inplace=True)
dft.sort_values(by=['pci'],ascending=False,inplace=True)
dft.reset_index(inplace=True,drop=True)
dft[0:10][['product_name','pci']]
#+END_SRC

#+RESULTS:
:RESULTS:
: \ndassda\n
#+begin_example
                              country_name  eci
0  South Georgia and South Sandwich Islds. 3.91
1                                    Japan 2.39
2                                  Germany 2.15
3                              Switzerland 2.06
4                                   Sweden 2.02
5                                  Finland 1.93
6                           United Kingdom 1.76
7                 United States of America 1.74
8                                  Austria 1.74
9                               San Marino 1.59
#+end_example
#+begin_example
                                        product_name  pci
0                         Silicones in primary forms 5.03
1  Prepared culture media for development of micr... 4.94
2  Lubricating preparations and those used in oil... 4.72
3  Photographic (including cinematographic) labor... 4.71
4  Machinery and mechanical appliances; having in... 4.56
5  Industrial or laboratory electric (including i... 4.50
6  Machines; for making up paper pulp, paper or p... 4.45
7  Tools for working in the hand, pneumatic or wi... 4.44
8  Signalling glassware and optical elements of g... 4.41
9            Halides and halide oxides of non-metals 4.40
#+end_example
:END:

** Complexity weighted by destination (example: Ukraine)

You can also calculate economic complexity by destination.

We did this to explore opportunities for Ukraine (to connect to European value chains):

https://growthlab.cid.harvard.edu/publications/assessing-ukraines-role-european-value-chains-gravity-equation-cum-economic

(Using the ECI by destination we found that highly complex products from Ukraine in the 2000s were typically destined for the Russian market, which was also one of the largest importers of products from Ukraine. The detoriation in relations with Russia led to a significant decline in exports there from 2011 onwards, resulting in Ukraine suffering from not only a quantitative but also a qualitative decline in exports).

Hidalgo and Hausmann (2009) calculate complexity of country $c$ as the average PCI of all products for which ${RCA}_{cp}>1$.

Below we define it as the weighted average PCI, where weights are given by the value of country $c$’s exports in each product. This allows us to define an ECI for separate export markets.

Let $\mathcal{M}$ be the set of countries that together constitute an export market (say, the EU's Single Market). Now, the destination-market specific ECI for country $c$ is defined as:

\begin{equation} \label{e_ECI}
ECI_{c}^{\mathcal{M}}=\sum \limits_{p} \frac{\sum \limits_{d \in \mathcal{M}} X^{d}_{op}}{\sum \limits_{d \in \mathcal{M}} X^{d}_{o}} {PCI}_{p}   
\end{equation}

where $X_{op}^{d}$ represents the exports of product $p$ from exporter $o$ to importer $d$ and an omitted subscript indicates the summation over the omitted category: $X_{o}^{d}=\sum \limits_{p} X_{op}^{d}$.

To calculate this, we need a dataset that has country exports *per destination* for this, which is available in the Growth Lab's DataVerse as:

#+BEGIN_EXAMPLE
"country_partner_hsproduct4digit_years_2000_2016.csv"
#+END_EXAMPLE

As this file above is 16 gigabytes, we will load a version of it for only Ukraine's exports. This file has been processed outside of Google colab using the code below:

#+BEGIN_EXAMPLE
df = pd.read_csv('country_partner_hsproduct4digit_years_2000_2016.csv')
df = df[df['location_code_code']=='UKR')
df = df[df['export_value']>0]
df.to_csv('ukr_exports_per_destination.csv',index=False)
#+END_EXAMPLE

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
print(f'loading data (from dropbox)')
df_ukr = pd.read_csv('https://www.dropbox.com/s/megm8qzn3jcwnqz/ukr_exports_per_destination.csv?dl=1')
print('loaded')

# show 10 random rows
df_ukr.sample(n=10)

#+END_SRC

Merge PCI from products in 2000 into the dataframe (from df_ec created in previous section using py-ecomplexity).

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

# -- to merge, add leading zeroes to dataset, e.g. 303 will be 0303
df_ukr.loc[ (df_ukr['hs_product_code'].astype(str).str.len()==3), 'hs_product_code']= '0'+ df_ukr['hs_product_code'].astype(str)
# -- remove leading / trailing spaces
df_ukr['hs_product_code'] = df_ukr['hs_product_code'].astype(str).str.strip()
# -- keep pcis from products in 2000
df_pci = df_ec[df_ec['year']==2000][['product_code','pci']].drop_duplicates(subset='product_code')
# -- merge pcis into the dataframe
df_ukr = pd.merge(df_ukr,df_pci[['product_code','pci']],how='left',left_on=f'hs_product_code',right_on=f'product_code',indicator=True)
# check merge 'left-only','right-only','both' counts (always do this to cross-check the merge)
df_ukr['_merge'].value_counts()
#+END_SRC


Now we calculate the ECI by destination.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

def calc_ecimc(data,origin_col,destination_col,product_col,value_col,pci_col):
    """
    Calculates economic complexity by destination (the weighted-by-exports-to-destination average PCI).

    Needs a year-origin-destination-product-exportvalue-pci dataframe as input.

    Returns:
        pandas dataframe with ecim
    """
    dft = data.copy()
    dft['export_value_cot'] = dft.groupby([origin_col,destination_col])[value_col].transform('sum')
    dft['pci_x_export'] = dft[pci_col] * dft[value_col]
    dft['pci_x_export_sum'] = dft.groupby([origin_col,destination_col])['pci_x_export'].transform('sum')
    dft['eciMc'] = dft['pci_x_export_sum']/dft['export_value_cot']
    dft.drop_duplicates(subset=[origin_col,destination_col],inplace=True)
    dft = dft[[origin_col,destination_col,'eciMc']]

    return dft

df_ukr_ecimc = calc_ecimc(data=df_ukr,
                          origin_col='location_code',
                          destination_col= 'partner_code',
                          product_col='hs_product_code',
                          value_col='export_value',
                          pci_col = 'pci'
                        )

# Show 10 random rows
df_ukr_ecimc.sample(n=20)
#+END_SRC


*** Cross-check with R ::noexport::

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
df_ukr_ecimc.sort_values(by=['eciMc'],ascending=False,inplace=True)
df_ukr_ecimc.head()
#+END_SRC

*** Map
Map economic complexity of Ukraine's exports by destination.

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
path = geopandas.datasets.get_path('naturalearth_lowres')
world = geopandas.read_file(path)
# merge complexities into it
world = pd.merge(world,df_ukr_ecimc[df_ukr_ecimc['eciMc']<3],how='left',left_on=f'iso_a3',right_on='partner_code',indicator=True)
fig, ax = plt.subplots(1, 1,figsize=(15,15))
plt.axis('off')
# https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
cmap = cm.coolwarm
world.plot(column='eciMc', ax=ax, legend=True,cmap=cmap)
plt.show()
#+END_SRC

Highly complex products are typically destined for the Russian market, which is also one of the largest importers of products from Ukraine.

The detoriation in relations with Russia led to a significant decline in exports there from 2011 onwards:
[[https://www.dropbox.com/s/xfl3gig3zxer0fm/total_exports_Ukraine_over_years.png?dl=1]]

As a result, Ukraine suffers from not only a quantitative but also a qualitative decline in exports. In the paper we explore new opportunities for Ukraine.

(Note: double-check political controversies when using mapping libraries in Python / R (e.g. geopandas, highcharter)!)

[[https://www.dropbox.com/s/twtl8p5ksgfezm0/map_ukraine.png?dl=1]]


* ---------------- Break: Excercise 2 ------------------

** What are countries with high complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC

** Vice versa, what are countries with low complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC

** What are products (PCI) with high complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC

** Vice versa, what are products (PCI) with low complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC

** Ukraine

*** How did Ukraine's economic complexity evolve over time?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC

*** How does Ukraine's economic complexity in 2015 compare to other countries? Which countries have comparable economic complexity?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC

*** What are the most complex products that Ukraine exported in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
#+END_SRC



* ---------------------------------------------------------------
* ---------------------------------------------------------------
* ---------------------------------------------------------------

* Excercise answers

** Excercise 1

*** What product does Ukraine export most in 1995? (excluding services such as 'transport', 'ict' etc)

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
df2 = df_orig[ (df_orig['country_name']=='Ukraine') & (df_orig['year'] == 2005) ].copy()
df3 = df2.groupby(['product_code','product_name'],as_index=False)['export_value'].sum()
df3.sort_values(by=['export_value'],ascending=False,inplace=True)
df3[['product_name','export_value']][0:5]
#+END_SRC

*** What products is Ukraine specialized in in 1995 and 2005 and how much do they export of these?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :async yes
# 1995

# Use the 'df_rca' dataframe for this

df2 = df_rca[ (df_rca['year']==1995) & (df_rca['country_name']=='Ukraine')].copy()
df2.sort_values(by=['RCAcpt'],ascending=False,inplace=True)
df2[['product_name','RCAcpt','year','export_value']][0:5]

# 2005
df2 = df_rca[ (df_rca['year']==2005) & (df_rca['country_name']=='Ukraine')].copy()
df2.sort_values(by=['RCAcpt'],ascending=False,inplace=True)
df2[['product_name','RCAcpt','year','export_value']][0:5]

#+END_SRC

*** Which product is most related to the product 'Stainless steel wire'?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes
PRODUCT = 'Stainless steel wire'
# select only this product
dft = df_cppt[df_cppt['product_name_1']==PRODUCT].copy()
# sort from high to low on phi
dft.sort_values(by=['phi'],ascending=False,inplace=True)
# show only first row
dft[0:1]
#+END_SRC

*** Plot Ukraine in the product space in 1995.

How would you characterize Ukraine's position in the product space?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

# Select country
COUNTRY_STRING = 'Ukraine'
df_ps = df_rca[df_rca['country_name']==COUNTRY_STRING].copy()

# Cross-check
if df_ps.shape[0] == 0:
    print('Country string set above does not exist in data, typed correctly?')
    STOP

# Select year
df_ps = df_ps[df_ps['year']==1995].copy()
#df_ps = df_ps[df_ps['year']==2005].copy()

# Select RCA > 1
df_ps = df_ps[df_ps['RCAcpt']>1]

# Keep only relevant columns
df_ps = df_ps[['product_name','export_value']]

# Keep only products with minimum value threshold
exports_min_threshold = 40000000
df_ps = df_ps[df_ps['export_value']>exports_min_threshold]

# Show resulting dataframe
df_ps.sample(n=5)

# And finally plot in the product space
create_product_space(df_plot_dataframe=df_ps,
                     df_plot_node_col='product_code',
                     df_node_size_col='export_value')
print('plotted')
#+END_SRC

*** Plot Ukraine in the product space in 2015.

Do you notice a difference with 1995?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py" :results output file :async yes

# Select country
COUNTRY_STRING = 'Ukraine'
df_ps = df_rca[df_rca['country_name']==COUNTRY_STRING].copy()

# Cross-check
if df_ps.shape[0] == 0:
    print('Country string set above does not exist in data, typed correctly?')
    STOP

# Select year
df_ps = df_ps[df_ps['year']==2015].copy()
#df_ps = df_ps[df_ps['year']==2005].copy()

# Select RCA > 1
df_ps = df_ps[df_ps['RCAcpt']>1]

# Keep only relevant columns
df_ps = df_ps[['product_name','export_value']]

# Keep only products with minimum value threshold
exports_min_threshold = 40000000
df_ps = df_ps[df_ps['export_value']>exports_min_threshold]

# Show resulting dataframe
df_ps.sample(n=5)

# And finally plot in the product space
create_product_space(df_plot_dataframe=df_ps,
                     df_plot_node_col='product_code',
                     df_node_size_col='export_value',)
print('plotted')
#+END_SRC


*** Plot your own country across different years in the product space. Do the results make sense? Do you notice any patterns?

** Excercise 2:

*** What are countries with high complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
qt_high = df_ec[df_ec['year']==2015]['eci'].quantile(0.95)
df_ec[df_ec['eci']>qt_high][['country_name']].drop_duplicates()[0:10]
#+END_SRC

*** Vice versa, what are countries with low complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
qt_low = df_ec[df_ec['year']==2015]['eci'].quantile(0.05)
df_ec[df_ec['eci']<qt_low][['country_name']].drop_duplicates()[0:10]
#+END_SRC

*** What are products (PCI) with high complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
qt_high = df_ec[df_ec['year']==2015]['pci'].quantile(0.95)
df_ec[df_ec['pci']>qt_high][['product_name']].drop_duplicates()[0:10]
#+END_SRC

*** Vice versa, what are products (PCI) with low complexity in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
qt_low = df_ec[df_ec['year']==2015]['pci'].quantile(0.05)
df_ec[df_ec['pci']<qt_low][['product_name','pci']].drop_duplicates()[0:10]
#+END_SRC

*** Ukraine

**** How did Ukraine's economic complexity evolve over time?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
df = df_ec[df_ec['country_name']=='Ukraine']
# drop duplicates of products
df.drop_duplicates(subset=['country_name','year'],inplace=True)
# keep relevant columns
df = df[['country_name','year','eci']]
# sort by ECI
df.sort_values(by='year',ascending=False,inplace=True)
df.reset_index(inplace=True,drop=True)
df.plot(x='year', y='eci')

#+END_SRC

#+RESULTS:
:RESULTS:
: <AxesSubplot:xlabel='year'>
: 2016
[[file:/Users/admin/Dropbox/proj/org_za_jupyter_output/00dd430c5709df439f5070d068cfdf5f82be854d.png]]
:END:

**** How does Ukraine's economic complexity in 2015 compare to other countries? Which countries have comparable economic complexity?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
df = df_ec[df_ec['year']==2015].copy()
# drop duplicates of countries
df = df[['country_name','eci']].drop_duplicates()
# sort by ECI
df.sort_values(by='eci',ascending=False,inplace=True)
df.reset_index(inplace=True,drop=True)
# create rank variable
df['rank'] = df.index
# get rank of Ukraine
RANK_UKRAINE = df[df['country_name']=='Ukraine'].reset_index()['rank'][0]
# check countries ranked directly above and below Ukraine
df[ (df['rank']>RANK_UKRAINE-10) & (df['rank']<RANK_UKRAINE+10)]
#+END_SRC

**** What are the most complex products that Ukraine exported in 2015?

#+BEGIN_SRC jupyter-python :tangle "~/Dropbox/proj/prog/zz_pythontangle.py"  :async yes
df = df_ec[df_ec['country_name']=='Ukraine'].copy()
df = df[df['year']==2015]
df.sort_values(by=['pci'],ascending=False,inplace=True)
df.reset_index(inplace=True,drop=True)
df[0:10][['product_name','pci']]
#+END_SRC

#+RESULTS:
#+begin_example
                                        product_name  pci
0  Horsehair and horsehair waste; whether or not ... 5.49
1  Cooking or heating apparatus of a kind used fo... 5.33
2  Machine-tools; for working any material by rem... 5.31
3  Cermets; articles thereof, including waste and... 5.03
4  Chemical preparations for photographic uses (o... 4.96
5  Photographic (including cinematographic) labor... 4.82
6  Chemical elements doped for use in electronics... 4.73
7                            Artificial filament tow 4.68
8  Tin; tubes, pipes and tube or pipe fittings (e... 4.67
9  Machining centres, unit construction machines ... 4.60
#+end_example
