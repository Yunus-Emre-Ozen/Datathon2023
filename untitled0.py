# encoding:utf-8


import matplotlib as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing


train_data = pd.read_csv(r"C:/Users/YunusEmreÖzen/Desktop/Datathon2023/1.csv", encoding='UTF-8')
test_data = pd.read_csv(r"C:/Users/YunusEmreÖzen/Desktop/Datathon2023/2.csv", encoding='UTF-8')



test_data["Eğitim Düzeyi"] = test_data["Eğitim Düzeyi"].replace(['Yüksek Lisans Mezunu', 'Lise Mezunu', 'Ortaokul Mezunu',
       'Yüksekokul Mezunu', 'Üniversite Mezunu', 'Doktora Mezunu',
       'İlkokul Mezunu', 'Doktora Ötesi', 'Eğitimsiz'],[6,3,2,4,5,7,1,8,0])



test_data["Cinsiyet"] = test_data["Cinsiyet"].replace(["Kadın", "Erkek"], [0,1])

test_data["Medeni Durum"] = test_data["Medeni Durum"].replace(["Bekar", "Evli"], [0,1])
test_data["Eğitime Devam Etme Durumu"] = test_data["Eğitime Devam Etme Durumu"].replace(["Etmiyor","Ediyor"], [0,1])



test_data["Yaş Grubu"] = test_data["Yaş Grubu"].replace(["18-30","31-40","41-50","51-60",">60"],[0,1,2,3,4])

final_test_data = pd.get_dummies(test_data, columns = ["İstihdam Durumu", "İstihdam Durumu","Yaşadığı Şehir", "En Çok İlgilendiği Ürün Grubu"])

train_data["Eğitim Düzeyi"] = train_data["Eğitim Düzeyi"].replace(['Yüksek Lisans Mezunu', 'Lise Mezunu', 'Ortaokul Mezunu',
       'Yüksekokul Mezunu', 'Üniversite Mezunu', 'Doktora Mezunu',
       'İlkokul Mezunu', 'Doktora Ötesi', 'Eğitimsiz'],[6,3,2,4,5,7,1,8,0])

train_data["Yaş Grubu"] = train_data["Yaş Grubu"].replace(["18-30","31-40","41-50","51-60",">60"],[0,1,2,3,4])


train_data["Öbek İsmi"] = train_data["Öbek İsmi"].replace(["obek_1","obek_2","obek_3","obek_4","obek_5","obek_6","obek_7","obek_8"],[1,2,3,4,5,6,7,8])

train_data["Cinsiyet"] = train_data["Cinsiyet"].replace(["Kadın", "Erkek"], [0,1])

train_data["Medeni Durum"] = train_data["Medeni Durum"].replace(["Bekar", "Evli"], [0,1])

train_data["Eğitime Devam Etme Durumu"] = train_data["Eğitime Devam Etme Durumu"].replace(["Etmiyor","Ediyor"], [0,1])






final_train_data = pd.get_dummies(train_data, columns = ["İstihdam Durumu", "İstihdam Durumu","Yaşadığı Şehir", "En Çok İlgilendiği Ürün Grubu"])





X_train = final_train_data.drop("Öbek İsmi", axis = 1)
Y_train = final_train_data["Öbek İsmi"]


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=10000000000000, penalty='l2')
log_model.fit(X_train, Y_train)


#After logistic regression we would look into predictions

predictions = log_model.predict(final_test_data)

predictions = pd.Series(predictions)

predictions = predictions.replace([1,2,3,4,5,6,7,8],["obek_1","obek_2","obek_3","obek_4","obek_5","obek_6","obek_7","obek_8"])

