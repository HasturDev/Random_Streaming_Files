# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


# %%
data = pd.read_csv('obesity-cleaned.csv')


# %%
data.head(20)


# %%
data.info()


# %%
data['country'].unique()


# %%
data = pd.read_csv('obesity-cleaned.csv')

data.columns = (['Number', 'Country', 'Year', 'Obesity', 'Sex'])

data["Estimate"]= data["Obesity"].map(lambda x: (x.split(" ")[1]))

data["Estimate"]= data.Estimate.str.replace("[", " ")

data["Estimate"]= data.Estimate.str.replace("]", " ")

data = data[data["Estimate"] != "data"]

data["Low_Estimate"]= data["Estimate"].map(lambda x: (x.split("-")[0])).apply(lambda x: float(x))

data["Obesity"]= data["Obesity"].map(lambda x: (x.split(" ")[0]))

data = data[data["Obesity"] != "No"]

data['Obesity']= data['Obesity'].apply(lambda x: float(x))

data["High_Estimate"]= data["Estimate"].map(lambda x: (x.split("-")[1])).apply(lambda x: float(x))

data.head(6)


# %%
print(data.isnull().sum())


# %%
sns.heatmap(data.isnull()).set(title = 'Missing Data', xlabel = 'Columns', ylabel = 'Data Points')


# %%
g = sns.pairplot(data[['Country','Year', 'Obesity', 'Sex', 'Low_Estimate', 'High_Estimate']])
g.fig.suptitle('Correlations')
plt.show()


# %%
fig = plt.figure(figsize=(100,100))
sns.jointplot(data= data, x= "Obesity", y= 'Year', kind= 'hex', color='coral')
plt.title('Some fat people')
plt.show()


# %%

fig = plt.figure(figsize=(16,8))
 
top_obese_countries = data[(data["Year"]==2016) & (data["Sex"]=="Both sexes") & (data["Obesity"] > 33)].groupby("Country").Obesity.sum().sort_values(ascending=False)
top_obese_countries.plot(kind="bar",title='Countries with 1/3 of population Obese in 2016', fontsize=20)


# %%
fig = plt.figure(figsize=(120, 4))

data.groupby('Country')["Obesity"].mean().sort_values().plot(kind='bar', color='coral')

plt.title('Average Obesity per Country')

plt.xlabel("Country")

plt.ylabel('Obesity')

plt.show()


# %%
rcParams['figure.figsize'] = 16, 8
fig = plt.figure()
plt.subplot(2, 2, 1)
countries = ["United States of America","Mexico","Canada","Guatemala","Cuba"]
for country in countries:
    plt.plot(data[(data["Country"]==country) & (data["Sex"]=="Both sexes")].Year,data[(data["Country"]==country) & (data["Sex"]=="Both sexes")].Obesity, label=country )
plt.xlabel('Year', fontsize=20)
plt.title('North Americas ', fontsize=20)
plt.style.use('fivethirtyeight')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.subplot(2, 2, 2)
countries = ["Brazil","Colombia","Argentina","Peru","Venezuela","Chile","Paraguay","Uruguay"]
for country in countries:
    plt.plot(data[(data["Country"]==country) & (data["Sex"]=="Both sexes")].Year,data[(data["Country"]==country) & (data["Sex"]=="Both sexes")].Obesity, label=country )
plt.xlabel('Year', fontsize=20)
plt.title('South Americas ', fontsize=20)
plt.style.use('fivethirtyeight')
plt.grid(True)
plt.legend()
plt.tight_layout()


# %%
rcParams['figure.figsize'] = 16, 8
all_sexes = data[data["Sex"]=="Both sexes"].groupby("Year").Obesity.mean()
male = data[data["Sex"]=="Male"].groupby("Year").Obesity.mean()
female = data[data["Sex"]=="Female"].groupby("Year").Obesity.mean()
plt.plot(all_sexes,linestyle='solid',marker='o',label="Obesity% of both Sexes")
plt.plot(male,linestyle='dashed',marker='.',label="Obesity% of male")
plt.plot(female,linestyle='dashdot',marker='^',label="Obesity% female")
plt.xlabel('Year', fontsize=20)
plt.ylabel('Obesity%', fontsize=20)
plt.title('Mean Obesity by Year', fontsize=20)
plt.style.use('fivethirtyeight')
plt.grid(True)
plt.legend()
plt.tight_layout()


# %%
px.scatter(data[data.Sex=='Both sexes'],x='Low_Estimate',y='High_Estimate',size='Obesity',
          animation_frame='Year',animation_group='Country',hover_data={"Obesity":True,"Low_Estimate":True,'Year':False},
          color='Country',hover_name='Country', range_y=[0,data[data.Sex=='Both sexes'].High_Estimate.max()],
           range_x=[0,data[data.Sex=='Both sexes'].Low_Estimate.max()])


# %%



