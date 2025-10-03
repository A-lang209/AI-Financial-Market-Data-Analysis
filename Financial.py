import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('/content/ai_financial_market_daily_realistic_synthetic.csv')
df.info()

#Convert datatype of Date column into DataTime format
df['Date'] = pd.to_datetime(df['Date'])
df.info()

df.head()
df['Company'].unique()

# Create a new column for 'Year' only
df['Year'] = df['Date'].dt.year
df.head()
df

df['Year'].unique() 
df['Year'].nunique()
df

df['Event'].value_counts()
df[df['Event'] == 'GPT-4 release']

df.isnull().sum()

df.head()

RD_s = df.groupby('Company')['R&D_Spending_USD_Mn'].sum()/1000
RD_s
plt.bar(RD_s.index, RD_s.values, color = ['cyan', 'black', 'magenta'])
plt.title( "R&D Spending by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")
plt.show()

df.head()

REV_c = df.groupby('Company')['AI_Revenue_USD_Mn'].sum()/1000
REV_c
plt.bar(REV_c.index, REV_c.values, color = ['cyan', 'black', 'magenta'], width=0.2)
plt.title( "Revenue earned by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")
plt.show()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(RD_s.index, RD_s.values, color = ['cyan', 'black', 'magenta'])
plt.title( "R&D Spending by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")
plt.subplot(1,2,2)
plt.bar(REV_c.index, REV_c.values, color = ['cyan', 'black', 'magenta'], width=0.2)
plt.title( "Revenue earned by the companies")
plt.xlabel("Company")
plt.ylabel("Amount in USD_$Bn")
plt.show()

df.head()

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Stock_Impact_%'], color='green')
plt.title("Change in Stock value")
plt.xlabel("Date ('Year')")
plt.ylabel("Stock_Impact_%")
plt.show()

df.head()

data_openai = df[df['Company'] == 'OpenAI']
data_openai

data_google = df[df['Company'] == 'Google']
data_google

data_meta = df[df['Company'] == 'Meta']
data_meta

plt.figure(figsize=(10,5))
plt.plot(data_openai['Date'], data_openai['Stock_Impact_%'], color='m')
plt.title("Change in Stock value of OpenAI")
plt.xlabel("Date")
plt.ylabel("Stock_Impact_%")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data_google['Date'], data_google['Stock_Impact_%'], color='c')
plt.title("Change in Stock value of Google")
plt.xlabel("Date")
plt.ylabel("Stock_Impact_%")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data_meta['Date'], data_meta['Stock_Impact_%'], color='black')
plt.title("Change in Stock value of Meta")
plt.xlabel("Date")
plt.ylabel("Stock_Impact_%")
plt.show()

data_openai.head()

data_openai.sort_values(by = 'Stock_Impact_%', ascending = False)
data_google.sort_values(by = 'Stock_Impact_%', ascending = False)
data_meta.sort_values(by = 'Stock_Impact_%', ascending = False)

df.head()

plt.figure(figsize=(10,5))
sns.scatterplot(x = 'Date', y = 'AI_Revenue_Growth_%', data = df, hue='Company')
plt.show()

df.sort_values(by = ['AI_Revenue_Growth_%'], ascending = False)

plt.plot(data_openai['Date'], data_openai['AI_Revenue_Growth_%'], color='m')
plt.show()

plt.plot(data_google['Date'], data_google['AI_Revenue_Growth_%'], color='c')
plt.show()

plt.plot(data_meta['Date'], data_meta['AI_Revenue_Growth_%'], color='black')
plt.show()

sns.heatmap(df.corr(numeric_only=True))

df.head()

spend = df.groupby('Year')['R&D_Spending_USD_Mn'].sum()
spend

plt.plot(spend.index, spend.values, color='r')
plt.title("Combined R&D Spending Year-by-Year")
plt.xlabel("Year")
plt.ylabel("Amount in USD_$Mn")
plt.show()

revenue  = df.groupby('Year')['AI_Revenue_USD_Mn'].sum()
revenue

plt.plot(revenue.index, revenue.values, color='g')
plt.title("Combined Revenue Earned Year-by-Year")
plt.xlabel("Year")
plt.ylabel("Amount in USD_$Mn")
plt.show()

plt.plot(spend.index, spend.values, color='r')
plt.plot(revenue.index, revenue.values, color='g')
plt.title( "Combined Expenditure vs Revenue Year-by-Year", fontsize = 12)
plt.xlabel("Year")
plt.ylabel("Amount in USD_$Mn")
plt.legend(['Expenditure', 'Revenue'])
plt.show()

# Pairplot to show the relations between the columns
sns.pairplot(df);

df.Event.value_counts()

df[df.Event == 'TensorFlow open-source release']

tf = df.loc[3955 : 3975]
tf

plt.figure(figsize=(10,4))
plt.plot(tf['Date'], tf['Stock_Impact_%'], color = 'c')
plt.title("Comparison before and after the release of TensorFlow open-source")
plt.xlabel("Date")
plt.ylabel("Change in Stock %")
plt.show()

df[ df.Event == 'GPT-4 release']

gpt4 = df.loc[ 2984 : 3004]
gpt4

plt.figure(figsize = (10,4))
plt.plot( gpt4['Date'], gpt4['Stock_Impact_%'], color = 'm')
plt.title("Comparison before and after the release of GPT-4")
plt.xlabel("Date")
plt.ylabel("Change in Stock %")
plt.show()

df.head(2)

df.groupby('Company')['Stock_Impact_%'].mean()*100
df.groupby('Company')['R&D_Spending_USD_Mn'].mean()
df.groupby('Company')['Stock_Impact_%'].max()

df.head(2)

stocks = df.groupby(['Year', 'Company'])['Stock_Impact_%'].max()
stocks

stocks.plot(kind = 'barh', color = ['r', 'black', 'm'])
plt.title("change in index")
plt.show()
Stock value") plt.xlabel("Year") plt.ylabel("Company
