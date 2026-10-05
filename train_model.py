import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
data=load_breast_cancer(); X,y=data.data,data.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
pipe=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=5000,random_state=42))])
grid=GridSearchCV(pipe,{"model__C":[.01,.1,1,10,100]},cv=5,scoring="accuracy",n_jobs=-1); grid.fit(Xtr,ytr)
p=grid.predict(Xte); print("Best parameters:",grid.best_params_); print("Accuracy:",accuracy_score(yte,p)); print(classification_report(yte,p,target_names=data.target_names)); print("ROC-AUC:",roc_auc_score(yte,grid.predict_proba(Xte)[:,1])); joblib.dump(grid.best_estimator_,"healthcare_diagnosis_model.joblib")
