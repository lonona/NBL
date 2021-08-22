'''
This Python script contains utility functions to perform
Stack Ensembling.
Produced date:: <Tuesday 11 August 2021>
'''

# << Import librairies >>
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
# import xgboost as xgb
# import lightgbm as lgb

from keras.models import Sequential
from keras.layers import Dense, Dropout

#Validation function
# from sklearn.preprocessing import StandardScaler
n_folds = 5
random_state = 42
n_jobs = -1

# get a list of models to evaluate
def get_basemodels():
	model = dict()
	model['LR'] = LinearRegression()
	model['lasso'] = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state= random_state))
	model['elastic_net'] = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio= .9, random_state= random_state))
	model['kernel_ridge'] = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
	model['knn'] = KNeighborsRegressor()
	model['cart'] = DecisionTreeRegressor()
	model['svm'] = SVR()
	# model['stack'] = get_stacking()
	
	return model


def data_preprocess(X,y):
	'''Split data into train and test form'''

	X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=0)
	
	
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

	# scaler.fit(X_train)

	# Now apply the transformations to the data:
	train_scaled = scaler.fit_transform(X_train)
	test_scaled = scaler.transform(X_test)

	return(train_scaled, test_scaled, y_train, y_test)


def base_rmse(model,xtrain,ytrain,xtest,ytest):
	
	model.fit(xtrain, ytrain)
	ypred = model.predict(xtest)
	
	return mean_squared_error(ytest, ypred, squared= False)

def rmsle_cv(model, X, y):
	kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
	rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
	
	return(rmse)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, models):
		self.models = models
		
	# we define clones of the original models to fit the data in
	def fit(self, X, y):
		self.models_ = [clone(x) for x in self.models]
		
		# Train cloned base models
		for model in self.models_:
			model.fit(X, y)

		return self
	
	#Now we do the predictions for cloned models and average them
	def predict(self, X):
		predictions = np.column_stack([
			model.predict(X) for model in self.models_
		])
		return np.mean(predictions, axis=1) 


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
	
	def __init__(self, base_models, meta_model, param_grid={}, n_folds=5):
		self.base_models = base_models
		self.meta_model = meta_model
		self.n_folds = n_folds
		self.param_grid = param_grid
   
	# We again fit the data on clones of the original models
	def fit(self, X, y):
		self.base_models_ = [list() for x in self.base_models]
		self.meta_model_ = clone(self.meta_model)

		kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)
		
		# Train cloned base models then create out-of-fold predictions
		# that are needed to train the cloned meta-model
		out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
		
		for i, model in enumerate(self.base_models):
			for train_index, holdout_index in kfold.split(X, y):
				try:
					instance = clone(model)
				except:
					instance = model
				self.base_models_[i].append(instance)
				
				instance.fit(X[train_index], y[train_index])				
				y_pred = instance.predict(X[holdout_index])
				out_of_fold_predictions[holdout_index, i] = y_pred
				
		# Fine tune meta_model hyperparameters
		self.meta_model_ = GridSearchCV(estimator=self.meta_model_, 
										param_grid=self.param_grid,
										cv=self.n_folds,
										scoring='neg_mean_squared_error', 
										verbose=1, n_jobs=-1)
		
		# Now train the cloned  meta-model using the out-of-fold predictions as new feature
		self.meta_model_.fit(out_of_fold_predictions, y)
		print("best parameters are: ", self.meta_model_.best_params_)
		
		return self
   
	#Do the predictions of all base models on the test data and use the averaged predictions as 
	#meta-features for the final prediction which is done by the meta-model
	def predict(self, X):
		meta_features = np.column_stack([
			np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
			for base_models in self.base_models_ ])
		return self.meta_model_.predict(meta_features)

def get_model(trainX, iterr=0):

	if iterr == 0:

		# define model 1
		model = Sequential()
		model.add(Dense(50, input_dim=trainX.shape[-1], activation='relu'))
		model.add(Dense(1, activation='softplus'))
		model.compile(loss='mse', optimizer='nadam') 
					  # metrics=['mae','mse','mape','CosineSimilarity','msle'])
		
	elif iterr == 1:
		# define model 2
		model = Sequential()
		model.add(Dense(50, input_dim=trainX.shape[-1], activation='relu'))
		model.add(Dense(25, activation='relu'))
		model.add(Dense(1, activation='softplus'))
		model.compile(loss='mse', optimizer='nadam')
					  # metrics=['mae','mse','mape','CosineSimilarity','msle'])
		
	
	elif iterr == 2:
		# define model 3
		model = Sequential()
		model.add(Dense(50, input_dim=trainX.shape[-1], activation='relu'))
		model.add(Dense(25, activation='relu'))
		model.add(Dense(15, activation = 'relu'))
		model.add(Dense(1, activation='softplus'))
		model.compile(loss='mse', optimizer='nadam')
		# metrics=['mae','mse','mape','CosineSimilarity','msle'])
		
	elif iterr == 3:
		# define model 4
		model = Sequential()
		model.add(Dense(50, input_dim=trainX.shape[-1], activation='relu'))
		model.add(Dense(25, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(10,activation = 'relu'))
		model.add(Dense(1, activation='softplus'))
		model.compile(loss='mse', optimizer='nadam')
		# metrics=['mae','mse','mape','CosineSimilarity','msle'])

	else:
		# define model 5
		model = Sequential()
		model.add(Dense(30, input_dim=trainX.shape[-1], activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(20, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10,activation = 'relu'))
		model.add(Dense(1, activation='softplus'))
		model.compile(loss='mse', optimizer='rmsprop')
	
	return model


def stacked_dataset(members, inputX):
	"""create stacked model input dataset as outputs from the ensemble"""

	predict_base_m = []
	stackX = None
	for model in members:
		
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		predict_base_m.append(yhat)
		
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.dstack((stackX, yhat))
	
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	
	return stackX, np.array(predict_base_m)


def fit_stacked_model(members, inputX, inputy, meta_learner, param_grid = {}):

	"""fit a model based on the outputs from the ensemble members"""

	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)[0]

	# fit standalone model
	meta_reg = GridSearchCV(estimator=meta_learner,
							 param_grid = param_grid,
							 cv=n_folds,
							 scoring='neg_mean_squared_error', 
							 verbose=1, 
							 n_jobs=n_jobs)

	meta_reg.fit(stackedX, inputy)

	#print the best parameters from all possible combinations
	print("best parameters are: ", meta_reg.best_params_)    
	
	return meta_reg


def stacked_prediction(members, model_learner, inputX):
	"""make a prediction with the stacked model"""

	# create dataset using ensemble
	stackedX, per_modl = stacked_dataset(members, inputX)
	
	# make a prediction
	yhat = model_learner.predict(stackedX)
	
	return yhat, per_modl[...,0]	