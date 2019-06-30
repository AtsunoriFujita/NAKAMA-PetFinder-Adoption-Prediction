# NAKAMA PetFinder Adoption Prediction


This respository contains our code for competition in kaggle.

27th Place Solution for [PetFinder Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction "PetFinder Adoption Prediction")

Team: [Y.Nakama](https://www.kaggle.com/yasufuminakama "Y.Nakama"),   [currypurin](https://www.kaggle.com/currypurin "currypurin"),   [atfujita](https://www.kaggle.com/atsunorifujita "atfujita"),   [copypaste](https://www.kaggle.com/copypaste0122 "copypaste")

Public score: 0.484(6th)    
Private score: 0.43455(27th)


### About nakama's feature
* Features from json files and text are almost same as public kernels
* Demographics features of Malaysia - GDP, Area, Population, HDI(Human Development Index)
* Image features extraction by Densenet121
* Var aggregation on basis of RescuerID to tell the model that if the RescuerID-Group treat their pets in the same way or not
* New health features of how many 1(good) or 3(Not Sure) in ['Health', 'Vaccinated', 'Dewormed', 'Sterilized']
* New age feature that expresses if the pet is younger or older in its RescuerID-Group or overall by using 'Age' and 'RescuerID_Age_var'


### curry's feature
The following features have high importance
* First image features extraction by Densenet121 and MobileNet
* second later image features extraction by Densenet121
* groupby RescuerID


### atfujita's feature
* pure_breed(x)
* image data SVD
* groupby RescuerID


### Ensemble method
* We performed ridge regression stacking using 9models(all GBDT).


### Blog
To check a part of our challenges, see this [blog](https://nmaviv.hatenablog.com/entry/2019/04/10/233211) (written in Japanese).
