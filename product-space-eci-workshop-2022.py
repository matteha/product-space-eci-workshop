# Install ecomplexity: pip3 install ecomplexity in terminal
# !pip3 install ecomplexity

# import os
# import sys

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"  # Show all output in Jupyter kernel

import pandas as pd

pd.set_option("display.max_columns", 500)  # Broaden pandas display in jupyter console
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 50)

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid") # White background plots: works on dark background

from ecomplexity import ecomplexity
# from ecomplexity import proximity

# Load trade data
df = pd.read_parquet("/n/hausmann_lab/lab/IPUMS/moved/temp3/trade.parquet")
# df = pd.read_parquet('/Users/admin/Dropbox/trade.parquet')
print("trade data loaded")

df.sample(n=10)

# keep 5 years
years = [1995, 2000, 2005, 2010, 2015]
df = df[df["year"].isin(years)]

# To use py-ecomplexity, specify the following columns
trade_cols = {
    "time": "year",
    "loc": "country_name",
    "prod": "product_name",
    "val": "export_value",
}

# Run ecomplexity
# NOTE : documentation
df2 = ecomplexity(df, trade_cols)

# Show results
df2.sample(n=10)

# Total exports per country/product per year
df2["exports_cy"] = df2.groupby(["country_name", "year"])["export_value"].transform(
    "sum"
)

df2["exports_py"] = df2.groupby(["product_name", "year"])["export_value"].transform(
    "sum"
)

# NOTE: What are countries with high complexity in 2015?
qt_high = df2[df2["year"] == 2015]["eci"].quantile(0.95)  # 95th percentile
df2[(df2["eci"] > qt_high) & (df2["exports_cy"] > 40000000) & (df2["year"] == 2015)][
    ["country_name"]
].drop_duplicates()[0:10]

# NOTE: Vice versa, what are countries with low complexity in 2015?

qt_low = df2[df2["year"] == 2015]["eci"].quantile(0.05)
df2[(df2["eci"] < qt_low) & (df2["exports_cy"] > 40000000) & (df2["year"] == 2015)][
    ["country_name"]
].drop_duplicates()[0:10]

# NOTE: What are products (PCI) with high complexity in 2015?

qt_high = df2[df2["year"] == 2015]["pci"].quantile(0.95)
df2[(df2["pci"] > qt_high) & (df2["exports_py"] > 10000000) & (df2["year"] == 2015)][
    ["product_name"]
].drop_duplicates()[0:10]

# NOTE: Vice versa, what are products (PCI) with low complexity in 2015?

qt_low = df2[df2["year"] == 2015]["pci"].quantile(0.05)
df2[(df2["pci"] < qt_low) & (df2["exports_py"] > 10000000) & (df2["year"] == 2015)][
    ["product_name"]
].drop_duplicates()[0:10]


# NOTE: Ukraine

# NOTE: How did Ukraine's economic complexity evolve over time?

dft = df2[df2["country_name"] == "Ukraine"].copy()

# drop duplicates of products
dft.drop_duplicates(subset=["country_name", "year"], inplace=True)

# keep relevant columns
dft = dft[["country_name", "year", "eci"]]

# sort by ECI
dft.sort_values(by="year", ascending=False, inplace=True)
dft.reset_index(inplace=True, drop=True)

# plot
fig = dft.plot(x="year", y="eci")
plt.show()

# NOTE: How does Ukraine's economic complexity in 2015 compare to other countries?
# NOTE: i.e. which countries have comparable economic complexity?

dft = df2[df2["year"] == 2015].copy()
# drop duplicates of countries
dft = dft[["country_name", "eci"]].drop_duplicates()
# sort by ECI
dft.sort_values(by="eci", ascending=False, inplace=True)
dft.reset_index(inplace=True, drop=True)
# create rank variable
dft["rank"] = dft.index
# get rank of Ukraine
RANK_UKRAINE = dft[dft["country_name"] == "Ukraine"].reset_index()["rank"][0]
# check countries ranked directly above and below Ukraine
dft[(dft["rank"] > RANK_UKRAINE - 10) & (dft["rank"] < RANK_UKRAINE + 10)]

# NOTE: What are the most complex products that Ukraine exported in 2010?

dft = df2[df2["country_name"] == "Ukraine"].copy()
dft = dft[dft["year"] == 2010]
dft.sort_values(by=["pci"], ascending=False, inplace=True)
dft.reset_index(inplace=True, drop=True)
dft[0:10][["product_name", "pci"]]

# NOTE: ... and in 2015?

dft = df2[df2["country_name"] == "Ukraine"].copy()
dft = dft[dft["year"] == 2015]
dft.sort_values(by=["pci"], ascending=False, inplace=True)
dft.reset_index(inplace=True, drop=True)
dft[0:10][["product_name", "pci"]]
