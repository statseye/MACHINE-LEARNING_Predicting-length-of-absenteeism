# Predicting length of absenteeism

Folder *source code* includes code in Python for predicting length of absenteeism at work using four ML techniques:
- poisson regression, 
- negative binomial regression,
- random forest, and 
- xgboost regressor. 

The data set comes from records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil. The target variable - number of asbent days at work - is clearly count data. Count variables have a boundary at 0 and are discrete, not continuous. The general advice is to analyze these with some variety of a Poisson model. I started with fitting Poisson regression model to the data but observed a fairly bad fit. Indeed, the data is over-dispersed, hence one of the assumptions of Poisson Models (that the conditional mean and the conditional variance are equal) is violated. This maked a very convincing case for using negative binomial regression. It can be considered as a generalization of Poisson regression since it has the same mean structure as Poisson regression and it has an extra parameter to model the over-dispersion. The model was a better fit to the data than Poisson, however I have not achieved satisfactory results. Hence, ensemble methods have been tried: random forest and xgboost regressor. Advanced parameter tunning resulted in random forest model achieving better prediciton results on test dataset, while xgboost regressor outperfomed all of the models tested. 

Output in Yupyter Notebook comming soon!
