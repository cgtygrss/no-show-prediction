import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

df_raw= pd.read_csv('C:\\Users\gcaga\Desktop\masaüstü\KaggleV2-May-2016.csv')
df_raw.info()
df_raw['PatientId']=df_raw['PatientId'].astype('int64')
df_raw['ScheduledDay'] = df_raw['ScheduledDay'].astype('datetime64[ns]')
df_raw['AppointmentDay'] = df_raw['AppointmentDay'].astype('datetime64[ns]')
df_raw.head()
df_raw.info()
df_raw.rename(columns={"Hipertension":"Hypertension","Handcap":"Handicap","SMS_received":"SMSReceived", "No-show":"NoShow"},inplace=True)

print(sorted(df_raw['Neighbourhood'].unique()))
print(sorted(df_raw['Age'].unique()))
df_raw[df_raw['Age'] ==-1]
df_raw[df_raw['Age'] == 115]
df_raw = df_raw[(df_raw['Age'] < 115) & (df_raw['Age'] > 0)]
df_raw = df_raw.drop(['PatientId','AppointmentID'],axis=1)
df_raw['ScheduledMonth'] = df_raw['ScheduledDay'].dt.month
df_raw['ScheduledDayofWeek'] = df_raw['ScheduledDay'].dt.day_name()
df_raw['ScheduledHour'] = df_raw['ScheduledDay'].dt.hour
df_raw['AppointmentMonth'] = df_raw['AppointmentDay'].dt.month
df_raw['AppointmentDayofWeek'] = df_raw['AppointmentDay'].dt.day_name()

df_raw['AppointmentHour'] = df_raw['AppointmentDay'].dt.hour
df_raw.info()
df_raw.head()
sns.countplot(x='Gender',hue = 'NoShow',data=df_raw)
sns.countplot(x='Age',hue = 'NoShow',data=df_raw)
sns.countplot(x='Scholarship',hue = 'NoShow',data=df_raw)
sns.countplot(x='Hypertension',hue = 'NoShow',data=df_raw)
sns.countplot(x='Handicap',hue = 'NoShow',data=df_raw)
df_raw.Handicap.value_counts()
sns.countplot(x='Diabetes',hue = 'NoShow',data=df_raw)
sns.countplot(x='Alcoholism',hue = 'NoShow',data=df_raw)
sns.countplot(x='SMSReceived',hue = 'NoShow',data=df_raw)
plt.figure(figsize=(30,12))
fig = sns.countplot(x='Neighbourhood',hue='NoShow',data=df_raw)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90);
df_raw.info()
df_corr = df_raw.drop(['ScheduledMonth','ScheduledDayofWeek','ScheduledHour','AppointmentMonth','AppointmentDayofWeek','AppointmentHour','Neighbourhood'],axis=1)
def zero_one(data, column):
    data[column].replace({data[column].unique()[0]:0, data[column].unique()[1]:1}, inplace=True)
    zero_one(df_corr, 'NoShow')
    zero_one(df_corr, 'Gender')

    sns.heatmap(df_corr.corr(), vmin=-0.9, vmax=0.9,cmap='coolwarm')

df_raw['AppointmentDayofWeek'] = df_raw['AppointmentDay'].dt.weekday

df_raw['ScheduledDayofWeek'] = df_raw['ScheduledDay'].dt.weekday

def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

df_raw = binary_encode(df_raw, 'Gender', positive_value='M')
df_raw = binary_encode(df_raw, 'NoShow', positive_value='Yes')
df_raw = onehot_encode(df_raw, 'Neighbourhood', prefix='N')

df = df_raw.drop(['AppointmentDay','ScheduledDay'],axis=1)
df

scaler = StandardScaler()
scaler.fit(df.drop(['NoShow'],axis=1))
scaled_features = scaler.transform(df.drop('NoShow',axis=1))
df_feat = pd.DataFrame(scaled_features)
df_feat.head()
X = df_feat
y = df['NoShow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_pred = dtree.predict(X_test)
print(classification_report(y_test,dtree_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, dtree_pred))

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train,y_train)
log_pred = logmodel.predict(X_test)
print(classification_report(y_test,log_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, log_pred))

rfc= RandomForestClassifier(n_estimators=100,verbose=5)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, rfc_pred))

kNeariest = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
kNeariest.fit(X_train, y_train)
kNeariest_pred = kNeariest.predict(X_test)
print(classification_report(y_test,kNeariest_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, kNeariest_pred))

naiveBayes = GaussianNB()
naiveBayes.fit(X_train, y_train)
naiveBayes_pred = naiveBayes.predict(X_test)
print(classification_report(y_test,naiveBayes_pred))
print("Confusion matrix:\n",confusion_matrix(y_test, naiveBayes_pred))