# Feature Selection with Nearest Neighbor Classifier

Machine Learning requires training a model to understand, given certain features, how to classify data. For example, when trying to determine if someone is of high risk of developing cancer, a very strong feature is whether the individual smokes often. Thus, we can determine that this person is of high risk. This is the purpose and use case for machine learning. Given lots of features, which are important and correctly classify the data and which are not useful.

The purpose of this Project, given to us by Dr. Eamonn Keogh as part of the course CS170: Introduction to Artificial Intelligence, is to take in data with many features and correctly determine which features are the strongest and can give us the highest accuracy of classifying the data. We were given a small data set with a few features, and a large data set with dozens of features, and tasked with finding the subset of the strongest features. To do this, we used the nearest neighbor algorithm and wrapped it using forward selection and backward elimination. I chose to code this project in C++ 11, just to use auto keyword, as it is the language I have used the most, and the code for this project can be found in this Github repository.

---

## Results

### Small Data Set

Below is Figure 1, which shows the result I received when running “CS170_Small_Data__97.txt”, which was my small data set, through forward selection to begin with. It had 6 features and 500 instances.

<img width="441" alt="Image" src="https://github.com/user-attachments/assets/06eb4674-3358-44a7-b627-a3447bfa5c4e" />

We first begin with no features, also known as the default rate, and this results in an accuracy of 82.60%. From here, we begin adding features that result in the best accuracy at that level. We see a small jump when feature {2} is added and another major jump when we add feature 1, thus the subset {2, 1} with an accuracy of 96.00%. From here, adding new features results in smaller decreases in accuracy until finally when we have all features we get an accuracy of 85.40%. Thus, from this, we can conclude that the best subset of features for the small data set is subset: {2, 1}. To confirm this, we will use backward elimination on the same data set shown below.

Below is Figure 2, which shows the result I received when running “CS170_Small_Data__97.txt”, through backward elimination.

<img width="496" alt="Image" src="https://github.com/user-attachments/assets/dba4e5cb-d140-48be-8f2a-ad71f6ecceeb" />

Since it is a small data set, we see that our accuracy with all features and no features is the same as above in forward selection, 85.40% and 82.60% respectively. This actually is shown at every subset. Every subset for backward elimination results in the same accuracy as forward selection. Thus, again we see that the best subset is: {1, 2} with an accuracy of 96.00%

** Conclusion for Small Data Set: ** After running both algorithms, I can conclude that features “1” and “2” are the strongest features for this data set. If I was asked my boss to deploy this model, I would choose the subset {1, 2} and would expect an accuracy of 96.00% for this model.

### Large Data Set

Below is Figure 3, which shows the result I received when running “CS170_Large_Data__12.txt”, which was my large data set, through forward selection to begin with. It had 40 features and 1000 instances.

<img width="514" alt="Image" src="https://github.com/user-attachments/assets/a8cb4985-4f94-4da3-9fd2-ee2c1d6698b2" />

Again, like for the small data set, we begin with the default rate or no feature subset and get an accuracy of 80.20%. Also, like before, we add features that result in the best accuracy at each level. We see an initial jump when feature {26} is added with an accuracy of 84.60%, and an even large jump when feature “38” is added in the subset {26, 38} with an accuracy of 96.60%. From there, adding features to the subset results in the accuracy lowering until finally, it reaches the subset with all 40 features with an accuracy of 68.60%. Thus, from this, we can conclude that the best subset of features for the large data set is subset: {26, 38}. Similarly, to confirm this, we will use backward elimination on the same data set shown below.

Below is Figure 4, which shows the result I received when running “CS170_Large_Data__12.txt”, which was my large data set, through backward elimination.

<img width="423" alt="Image" src="https://github.com/user-attachments/assets/52e57592-9c95-4720-8f6e-004876c53594" />

Again, like the small data set, the accuracy with all features is 68.60% and the accuracy with no features is 80.20%. However, towards the end we see that backward elimination finds a different optimal subset for the large data set, showing it is slightly less accurate than forward selection for this data set. It finds subset {26, 31} with an accuracy of 85.80%. Since it also finds feature “26” as a strong feature, we can conclude that feature “26” is without a doubt a strong feature.

** Conclusion for Large Data Set: ** After running both algorithms, I can conclude that features “26” and “38” are the strongest features for this data set. There is slight evidence for feature “31” being a useful feature, but it cannot be determined without further research. Therefore, if I was asked my boss to deploy this model, I would choose the subset {26, 38} and would expect an accuracy of 96.60% for this model.

---


## Usage

1. **Clone the repository**
```bash
git clone https://github.com/DavidA346/CS170-Project-2.git
cd CS170-Project-2
