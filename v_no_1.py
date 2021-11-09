import pandas as pd

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


from functions import *

'''import trenovacich dat'''
train_data = load_data('./data/spotify_train.csv')
'''dropovanie zbytocnych veci'''
train_data=train_data.drop_duplicates(subset=['name','artist_id'])

train_data = train_data.drop(['id', 'artist', 'artist_id', 'name', 'artist_followers',
                              'url', 'playlist_id', 'playlist_description', 'playlist_name', 'playlist_url', 'query'],axis=1)




'''import testovacich dat'''
test_data = load_data('./data/spotify_test.csv')
test_data = test_data.drop(['id', 'artist',  'artist_id', 'name', 'artist_followers',
                              'url', 'playlist_id', 'playlist_description', 'playlist_name', 'playlist_url', 'query'],axis=1)



'''kodovanie stringovych hodnot zaner a rok 'release_date', 'artist_genres'     '''
train_data['release_date']=pd.DatetimeIndex(train_data['release_date']).year
categorical_columns=['artist_genres']
label_encoder = LabelEncoder()
train_data[categorical_columns] = train_data[categorical_columns].apply(label_encoder.fit_transform)

test_data['release_date']=pd.DatetimeIndex(test_data['release_date']).year
categorical_columns=['artist_genres']
label_encoder = LabelEncoder()
test_data[categorical_columns] = test_data[categorical_columns].apply(label_encoder.fit_transform)

''' remove outliers'''
train_data=remove_outliers(train_data)
test_data=remove_outliers(test_data)
print(train_data.isnull().sum())

'''vyhodit loudness pred normalizaciou'''

train_loudness=train_data.loudness
train_data.drop('loudness',axis=1)

test_loudness=test_data.loudness
test_data.drop('loudness',axis=1)
''' scaler fit '''
scaler=StandardScaler()
scaler.fit(train_data)

'''skalovanie trenovacich'''
scaled_train=scaler.transform(train_data)
scaled_train=pd.DataFrame(scaled_train,columns=train_data.columns)
'''skalovanie testovacich dat'''
scaled_test=scaler.transform(test_data)
scaled_test=pd.DataFrame(scaled_test,columns=test_data.columns)

'''dolepenie loudness'''
scaled_train['loudness']=train_loudness
scaled_test['loudness']=test_loudness

'''splitovanie trenovacich dat na trenovacie a validacne
    v podstate vyuzite na to aby som netrenoval celu noc 40k vzoriek
'''
X_train=scaled_train.drop('loudness',axis=1)
y_train=scaled_train.loudness

#X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.24,random_state=42)
print('Size of dataset:')
print(len(X_train))


'''priprava testovacich dat'''
X_test=scaled_test.drop('loudness',axis=1)
y_test=scaled_test.loudness

''' SVM regresia na predpoved hlasitosti '''
reg=svm.SVR(kernel='rbf',gamma=0.01,C=100)
reg.fit(X_train,y_train)
prediction=reg.predict(X_train)



print('Trenovacia mnozina')
print('R2: %.2f' % r2_score(y_train,prediction))
print('MSE: %.2f' % mean_squared_error(y_train,prediction))

print('Cross validation na trenovacej mnozine')
scrs=cross_val_score(reg,X_train,y_train)
print(scrs)

reg.fit(X_test,y_test)
prediction=reg.predict(X_test)

print('Testovacia mnozina')
print('R2: %.2f' % r2_score(y_test,prediction))
print('MSE: %.2f' % mean_squared_error(y_test,prediction))
rezidual_graph(y_test,prediction)

print('Cross validation na testovacej mnozine')
scrs=cross_val_score(reg,X_test,y_test)
print(scrs)


'''grid search'''
#grisearch(X_train, y_train,y_test,X_test)

'''Ensemble (tea)bagging'''
print('Bagging on train data')
bagging_reg(X_train,y_train)
print('Bagging on test data')
bagging_reg(X_test,y_test)
'''Ensemble boosting'''
print('Boosting on train data')
boosting_reg(X_train,y_train)
print('Boosting on test data')
boosting_reg(X_test,y_test)