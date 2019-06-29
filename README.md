# NAKAMA PetFinder Adoption Prediction


This respository contains my code for competition in kaggle.

27th Place Solution for [PetFinder Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction "PetFinder Adoption Prediction")

Public score: 0.484(6th)    
Private score: 0.43455(27th)

#### About nakama's feature
<li> Features from json files and text are almost same as public kernels
<li> Demographics features of Malaysia - GDP, Area, Population, HDI(Human Development Index)
<li> Image features extraction by Densenet121
<li> Var aggregation on basis of RescuerID to tell the model that if the RescuerID-Group treat their pets in the same way or not
<li> New health features of how many 1(good) or 3(Not Sure) in ['Health', 'Vaccinated', 'Dewormed', 'Sterilized']
<li> New age feature that expresses if the pet is younger or older in its RescuerID-Group or overall by using 'Age' and 'RescuerID_Age_var'

#### Blog
To check a part of our challenges, see this [blog](https://nmaviv.hatenablog.com/entry/2019/04/10/233211) (written in Japanese).
