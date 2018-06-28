# Corporación Favorita Grocery Sales Forecasting
Mike Irvine - Module 1 Capstone Project

July 29, 2018

Galvanize Data Science Immersive - Denver

## Executive Summary
### This is a test of a tagline

## Context & Key Question:
### *Corporación Favorita*, a grocery chain in Ecuador, wants to reduce stockouts and improve sales with a more accurate sales forecasting model.

Sales forecasting is a critical planning tool for all grocers, as it enables them to to stock the appropriate inventory levels while maximizing sales. If grocers over-predict sales, they have extra inventory that might need to be sold at a discount or may spoil. If they under-predict sales, they have stockouts which lead to disappointed customers and lost revenue. A more accurate sales forecasting method will ensure the right amount of product is in the right store at the right time - leading to increased sales and reduced stockouts.

A grocery store chain in Ecuador named *Corporación Favorita* faced a similar sales forecasting challenge when they approached Kaggle to host a data science competition. The challenge to the Kaggle community in the fall of 2017 was to build a model that more accurately forecasts product sales. *Corporación Favorita* relied on subjective forecasting methods with very little data to back them up and very little automation to execute plans. They were excited to see how machine learning could improve sales forecasting, so they can stock just enough of the right products at the right time.

My module 1 capstone project is to take part in this challenge by building a linear regression model to predict unit sales for *Corporación Favorita*. 

## Data Source
### *Corporación Favorita* provided daily store level sales aggregated by product item number, along with supplemental datasets to enrich the transactional data.
*Corporación Favorita* provided Kaggle a rich data set for the competition, which included:
- Daily store level sales aggregated by product item number
- Product item details including class and whether the item is perishable
- Store location details including city, state, type and cluster
- Store transaction details including total transactions per day by store
- Ecuador holiday dates including national, regional and local holidays (holidays affect store sales)
- Daily oil prices (Ecuador's economy is oil dependent so oil prices may impact grocery store sales)

## Exploratory Data Analysis
### Given the enormous dataset (125M+ daily unit sales records), I selected a single year and product family to build a sales forecasting model.
An outline of each table is below. My general approach was to use the supplemental datasets (items, holidays, store, etc.) to enrich the transaction dataset, and then select a subset of the data for a particula year and item family to reduce the scope of the model. 
##### Transaction Dataset: 
- The main transaction dataset has 125M+ records spanning from Jan 1, 2013 to Aug 15, 2017
- A record is defined as the unit sales volume (not price) for a particular item number by store by day.
- Only includes 6 features: id, date, store_nbr, item_nbr, unit_sales, and onpromotion
#### Items Dataset:
- 4100 records, which represent 4100 individual items
- Features include: item_nbr, class, and perishable
- All features are categorical
#### Stores Dataset:
- 54 records, which represent 54 store locations
- Features include: store_nbr and cluster, both of which are categorical
#### Store Transactions Dataset:
- 100k+ records, which represent the total daily transactions by store by day from Jan 1, 2013 to Aug 15, 2017
- Features include: store_nbr and transactions
#### Holidays Dataset:
- 300+ records, which represent each holiday from Jan 1, 2013 to Aug 15, 2017
- Features include: date, type, locale, locale name and description
- The locale feature identifies if the holiday is local, regional or national
#### Oil Dataset:
- 1200+ records, which represent the daily oil price from Jan 1, 2013 to Aug 15, 2017
- Features include: daily oil price

#### Approach - *focus on a single year, single item family*:
Given the large dataset, I decided to reduce the scope and focus on a single year and item family. I selected the 'MEATS' item family from August 2015 - August 2016 to be my training and test dataset.

The items dataset included 33 item families:

- ['GROCERY I',
 'CLEANING',
 'BREAD/BAKERY',
 'DELI',
 'POULTRY',
 'EGGS',
 'PERSONAL CARE',
 'LINGERIE',
 'BEVERAGES',
 'AUTOMOTIVE',
 'DAIRY',
 'GROCERY II',
 'MEATS',
 'FROZEN FOODS',
 'HOME APPLIANCES',
 'SEAFOOD',
 'PREPARED FOODS',
 'LIQUOR,WINE,BEER',
 'BEAUTY',
 'HARDWARE',
 'LAWN AND GARDEN',
 'PRODUCE',
 'HOME AND KITCHEN II',
 'HOME AND KITCHEN I',
 'MAGAZINES',
 'HOME CARE',
 'PET SUPPLIES',
 'BABY CARE',
 'SCHOOL AND OFFICE SUPPLIES',
 'PLAYERS AND ELECTRONICS',
 'CELEBRATION',
 'LADIESWEAR',
 'BOOKS']

Only selecting the 'MEATS' item family reduced the datasize to ~500k records. Key descriptive statistics for the target variable, 'unit_sales' below:

|Stat |    Value 
|-------|----------------|
|count  |  574203.00 |
|mean |        11.94 |
|std    |      32.96 |
|min     |    -44.26  |
|25%      |     2.52  |
|50%       |    5.30  |
|75%      |    11.61  |
|max      |  5357.83  |

Because of some extreme outliers (12 data points > 1000 unit_sales) that are several standard deviations away from the mean, I decided to remove them from the dataset. The table and histogram below shows details for the unit_sales for the MEATS item family after the outliers are removed. The mean is now 9.46 unit sales per day per store, a standard deviation of ~12, and a max value of 77.86.

|Stat |    Value 
|-------|----------------|
|count |   563817.00 |
|mean   |       9.46 |
|std    |      11.97 |
|min    |       0.00 |
|25%    |       2.48 |
|50%    |       5.17 |
|75%    |     11.04 |
|max    |      77.86 |



## Feature Engineering
### 249 features were created after merging datasets to enrich the transaction data, and creating dummy variables for all the categorical features.

- Question trying to answer
- Data source
- EDA
- Feature Engineering
- Modeling
- Results
- Future work
- References

NOTE: without removing outliers, rmse is 26+ and r squared is only ~.25
