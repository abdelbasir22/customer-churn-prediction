import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns



dataset = pd.read_csv('../data/telco_customer_churn.csv')

df = dataset.copy()
df.head()




sns.countplot(x='Churn',data=df)
plt.title('Churn Distribution')
plt.show()

sns.histplot(df['tenure'], kde=True)
plt.title('tenure distribution')
plt.show()



sns.histplot(df['MonthlyCharges'], kde=True)
plt.title('Monthly Charges Distribution')
plt.show()


sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Churn VS Tenure')
plt.show()


sns.countplot( x='Contract',hue='Churn', data=df)
plt.title('Churn VS Contract')
plt.show()


sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Churn VS Monthly Charges')
plt.show()


df.drop('customerID',axis=1, inplace=True)
df['gender'] = df['gender'].map({'Male':1, 'Female':0})
cols = ['Partner', 'Dependents', 'PhoneService', 'Churn']
for col in cols:
    df[col] = df[col].map({'Yes':1, 'No':0}) 

df= pd.get_dummies(df,drop_first=True)

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

df['AvgCharges'] = df['TotalCharges'] / df['tenure']
bins = [0, 12, 24, 48, 72]
labels = ['0-1yr', '1-2yr', '2-4yr', '4-6yr']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)
df['LongTermContract'] = df['Contract_One year'] + df['Contract_Two year']


from sklearn.model_selection import train_test_split

X = df.drop('Churn',axis=1)
Y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

accuracy= accuracy_score(y_test, y_predict)
print(f"Accuracy = {accuracy}" )

confusion_Matrix = confusion_matrix(y_test, y_predict)
print(f"Confusion Matrix =\n {confusion_Matrix} \n", )

classification_Report = classification_report(y_test, y_predict)
print(f"Classification Report = \n {classification_Report}", )

y_prob = model.predict_proba(x_test)[:,1]
y_new= np.where(y_prob>.4,1,0)

model= LogisticRegression(class_weight='balanced', max_iter=1000)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_new))



from sklearn.metrics import roc_curve, roc_auc_score

# احتمالات التنبؤ

# حساب الـ ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# حساب AUC
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.show()

print("AUC Score:", auc_score)


optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Best Threshold:", optimal_threshold)



















