# Feature_Engineering
## Training a model
Training a model is the process by which an algorithm learns about input and output data, modifying internal parameters, to minimize prediction error.
### To be able to train a model: 
-we read the .csv file

-we delete the columns that do not help us from that .csv file
  ```python
  y=train["target"]
  X=train.drop("target",axis=1)
  X
  X=X.drop("era",axis=1)
  X=X.drop("data_type",axis=1)
  X

  good_cols=[col for col in X.columns if not col.startswith("target")]
 
  X=X[good_cols]
  X
  ```

  -we add (optionally) some columns 

  -we choose a model (xgboost, catboost, lightgbm)
  ```python
  model=xgb.XGBRegressor( n_estimators=100)
  
  ```
 -we train the model
   
  ```python
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
  ```
    




 
