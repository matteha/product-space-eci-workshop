
# Install ecomplexity: pip3 install ecomplexity in terminal
!pip3 install ecomplexity

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  # Show all output in Jupyter kernel

import polars as pl
import pandas as pd 
pd.set_option('display.max_columns', 500) # Broaden pandas display in jupyter console
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth',50)

from matplotlib import pyplot as plt

from ecomplexity import ecomplexity
from ecomplexity import proximity

# Load trade data
# df = pd.read_csv('/Users/admin/Dropbox/trade.csv')
# df['product_code'] = df['product_code'].astype(str)
# df.to_parquet('/Users/admin/Dropbox/trade.parquet')
df = pd.read_parquet('/Users/admin/Dropbox/trade.parquet')
df[0:10]

df.sample(n=10)

# keep 5 years 
years = [1995,2000,2005,2010,2015]
df = df[df['year'].isin(years)]

# To use py-ecomplexity, specify the following columns
trade_cols = {'time':'year',
              'loc':'country_name',
              'prod':'product_name',
              'val':'export_value'
             }

# Run ecomplexity
# NOTE : documentation
df2 = ecomplexity(df, trade_cols)

# Show results
df2.sample(n=10)

df2.sort_values(by=['pci'],ascending=False,inplace=True)
df2[0:10]
df.reset_index(inplace=True,drop=True)

# NOTE: What are countries with high complexity in 2015?

qt_high = df2[df2['year']==2015]['eci'].quantile(0.95) # 95th percentile
df2[df2['eci']>qt_high][['country_name']].drop_duplicates()[0:10]

# NOTE: Vice versa, what are countries with low complexity in 2015?

qt_low = df2[df2['year']==2015]['eci'].quantile(0.05)
df2[df2['eci']<qt_low][['country_name']].drop_duplicates()[0:10]

# NOTE: What are products (PCI) with high complexity in 2015?

qt_high = df2[df2['year']==2015]['pci'].quantile(0.95)
df2[df2['pci']>qt_high][['product_name']].drop_duplicates()[0:10]

# NOTE: Vice versa, what are products (PCI) with low complexity in 2015?

qt_low = df2[df2['year']==2015]['pci'].quantile(0.05)
df2[df2['pci']<qt_low][['product_name','pci']].drop_duplicates()[0:10]

# NOTE: Ukraine

# NOTE: How did Ukraine's economic complexity evolve over time?

dft = df2[df2['country_name']=='Ukraine']

# drop duplicates of products
dft.drop_duplicates(subset=['country_name','year'],inplace=True)

# keep relevant columns
dft = dft[['country_name','year','eci']]

# sort by ECI
dft.sort_values(by='year',ascending=False,inplace=True)
dft.reset_index(inplace=True,drop=True)

# plot
fig = dft.plot(x='year', y='eci')
plt.show()

# NOTE: How does Ukraine's economic complexity in 2015 compare to other countries? 
# NOTE: i.e. which countries have comparable economic complexity?

dft = df2[df2['year']==2015].copy()
# drop duplicates of countries
dft = dft[['country_name','eci']].drop_duplicates()
# sort by ECI
dft.sort_values(by='eci',ascending=False,inplace=True)
dft.reset_index(inplace=True,drop=True)
# create rank variable
dft['rank'] = dft.index
# get rank of Ukraine
RANK_UKRAINE = dft[dft['country_name']=='Ukraine'].reset_index()['rank'][0]
# check countries ranked directly above and below Ukraine
dft[ (dft['rank']>RANK_UKRAINE-10) & (dft['rank']<RANK_UKRAINE+10)]

#NOTE: What are the most complex products that Ukraine exported in 2010?

dft = df2[df2['country_name']=='Ukraine'].copy()
dft = dft[dft['year']==2010]
dft.sort_values(by=['pci'],ascending=False,inplace=True)
dft.reset_index(inplace=True,drop=True)
dft[0:10][['product_name','pci']]


