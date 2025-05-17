# Feature Selection with Nearest Neighbor Classifier

Machine Learning requires training a model to understand, given certain features, how to classify data. For example, when trying to determine if someone is of high risk of developing cancer, a very strong feature is whether the individual smokes often. Thus, we can determine that this person is of high risk. This is the purpose and use case for machine learning. Given lots of features, which are important and correctly classify the data and which are not useful.

The purpose of this Project, given to us by Dr. Eamonn Keogh as part of the course CS170: Introduction to Artificial Intelligence, is to take in data with many features and correctly determine which features are the strongest and can give us the highest accuracy of classifying the data. We were given a small data set with a few features, and a large data set with dozens of features, and tasked with finding the subset of the strongest features. To do this, we used the nearest neighbor algorithm and wrapped it using forward selection and backward elimination. I chose to code this project in C++ 11, just to use auto keyword, as it is the language I have used the most, and the code for this project can be found using the GitHub link given up above in the cover page.

---

## Results
Below is Figure 1, which shows the result I received when running “CS170_Small_Data__97.txt”, which was my small data set, through forward selection to begin with. It had 6 features and 500 instances.
<img width="441" alt="Image" src="https://github.com/user-attachments/assets/06eb4674-3358-44a7-b627-a3447bfa5c4e" />
We first begin with no features, also known as the default rate, and this results in an accuracy of 82.60%. From here, we begin adding features that result in the best accuracy at that level. We see a small jump when feature {2} is added and another major jump when we add feature 1, thus the subset {2, 1} with an accuracy of 96.00%. From here, adding new features results in smaller decreases in accuracy until finally when we have all features we get an accuracy of 85.40%. Thus, from this, we can conclude that the best subset of features for the small data set is subset: {2, 1}. To confirm this, we will use backward elimination on the same data set shown below.

---


## Usage

1. **Clone the repository**
```bash
git clone https://github.com/DavidA346/CS170-Project-2.git
cd CS170-Project-2
