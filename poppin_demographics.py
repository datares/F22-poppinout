#standard imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, date

#import dataframe
data = pd.read_csv("poppinuser.csv")
data = data.drop(labels=["disabledMessageCategories","notifications","bio","status","index","teamTitle","profileImageURL","displayName","followers","following","sessions"],axis=1)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# add age column converted from birthdate column
birthdate = []
for i in range(len(data['birthdate'])):
    birthdate.append(data['birthdate'][i][0:10])
    
data['birthdate']=birthdate

def age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

data['age'] = data['birthdate'].apply(age)
data = data[data.age != 0]

# add column of days since last sign on converted from lastSignon column
lastSignon = []
for i in range(len(data['lastSignon'])):
    lastSignon.append(data['lastSignon'][i][0:10])

data['lastSignon']=lastSignon

def daysSince(time):
    d0 = datetime.strptime(time, "%Y-%m-%d").date()
    d1 = date(2022,11,22) #date the dataset was received
    delta = d1 - d0
    return delta.days

data['daysSinceLast']=data['lastSignon'].apply(daysSince)

# add column of activity of users
# A is classified as users that have used the app within 21 days from when the dataset was received
# N is classified as users that have not used the app for more then 21 days from when the dataset was received
activity = []
for i in range(len(data)):
    if data['daysSinceLast'][i] > 21:
        activity.append("N")
    else:
        activity.append("A")

#add empty frequency and age_gender columns for later use
data['frequency']=0
data['age_gender'] = ""

# clean data and make sub-dataframes
# frequency of each age and gender 
df = data.groupby(["age","gender","age_gender"])["frequency"].count().to_frame().reset_index() 
for i in range(len(df)):
    df['age_gender'][i] = str(df['age'][i])+" "+str(df['gender'][i])

# frequency of active users based on age, gender, and days since last logon
df2 = data.groupby(['age','gender','age_gender','activity'])["frequency"].count().to_frame().reset_index()
for i in range(len(df2)):
    df2['age_gender'][i] = str(df2['age'][i])+" "+str(df2['gender'][i])
    
# VISUALIZATION 1
# depict scatterplot of frequency of age by gender
fig = plt.gcf()
fig.set_size_inches(18, 10)
sns.set_style("whitegrid")
p = sns.stripplot(x="age",
                  y="frequency",
                  hue="gender", 
                  data=df,
                  jitter=True,
                  s = 20,
                  alpha = 0.9,
                  palette ="pastel")

p.set_title("Frequency of Users by Age and Gender", fontsize = 20)
plt.xlabel("Age", fontsize = 15)
plt.ylabel("Number of Users", fontsize = 15)
plt.legend(title="Gender", fontsize=12)
plt.show()


# VISUALIZATION 2
#depict catplot of frequency of active/non-active users by age/gender
fig = plt.gcf()
fig.set_size_inches(20,20)
sns.set_theme(style="whitegrid")

g = sns.catplot(data=df2[0:48], 
                kind="bar",
                x="activity", 
                y="frequency", 
                hue="age_gender",
                palette="bright", 
                alpha=.6, 
                height=6,
                aspect=2)

g.set_axis_labels("Active/Not Active", "Frequency", fontsize=15)
g.legend.set_title("Age and Gender")
g.fig.suptitle("Frequency of Active/Non-active Users by Age and Gender", fontsize=20)

