#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

//Returns the class label as an integer
int classifyInstance(vector<double> instance) {
    return instance.at(0);
}

//Returns the euclidean distance of 2 instances/rows
double euclideanDistance(vector<double> currentObj, vector<double> neighbor) {
    //Euclidean Distance = sqrt((x1 - x2)^2 + ... (xn -xn)^2)

    //We need sum to later sqrt it
    double sum = 0;

    //Iterate through each feature and calculate the sum starting at 1 to exclude the class label
    for (int i = 1; i < currentObj.size(); ++i) {
        sum += pow((currentObj.at(i) - neighbor.at(i)), 2);
    }

    //Returns square distance/euclidean distance
    return sqrt(sum);
}

//Returns default rate for forward selection
float defaultRate(vector<vector <double> >& data) {
    //Stores count for both classes
    float class1 = 0.0 ;
    float class2 = 0.0;

    //Iterate through each row and keep track of the size of each class label
    for (int i = 0; i < data.size(); ++i) {
        vector<double> classifyRow = data.at(i);
        int currClass = classifyInstance(classifyRow);

        if (currClass == 1) {
            ++class1;
        }
        
        else{
            ++class2;
        }
    }

    //Return the default rate of the highest class label we find
    if (class1 > class2) {
        return (class1 / data.size());
    }
    else {
        return (class2 / data.size());
    }
}

//Accuracy function needed for forward selection
float leaveOneOutCrossValidation(vector<vector <double> >& data, int totalFeatures, vector <int> currFeatures, int potentialNewFeature) {
    //Count need to check how many instances we classified correctly
    float correctlyClassifiedCnt = 0;

    //Get default rate when no features are passed in
    if (potentialNewFeature == -1) {
        return defaultRate(data);
    }

    //Iterates through each row
    for (int i = 0; i < data.size(); ++i) {
        //We want to classify each object based on its features
        vector<double> classifyObject = data.at(i);

        //Returns the label of the current instance
        int currInstanceLabel = classifyInstance(data[i]);

        //Create a temp vector with potential new feature added in case we do not add it at the end
        vector<int> temporaryAddingNewFeature = currFeatures;
        temporaryAddingNewFeature.push_back(potentialNewFeature);

        //Sets all features not in the current feature set to 0
        for (int k = 1; k < totalFeatures + 1; ++k) {
            auto it = find(temporaryAddingNewFeature.begin(), temporaryAddingNewFeature.end(), k);

            if (it == temporaryAddingNewFeature.end()) {
                classifyObject.at(k) = 0;
            }
        }

        //Initially nearest neighbor is infinity away
        double nearestNeighborDist = numeric_limits<double>::infinity();
        double nearestNeighborLoc = numeric_limits<double>::infinity();

        //Need the class of the nearest neighbor to see if we classifies it correctly
        int labelOfNearestNeighbor;

        //Finds nearest neighbor by comparing to each other instance
        for (int j = 0; j < data.size(); ++j) {
            //Current instance does not check itself
            if (i != j) {
                //Neighbor we want to compare to
                vector<double> currNeighbor = data.at(j);

                //Sets all features not in the current feature set to 0
                for (int l = 1; l < totalFeatures + 1; ++l) {
                    auto it = find(temporaryAddingNewFeature.begin(), temporaryAddingNewFeature.end(), l);

                    if (it == temporaryAddingNewFeature.end()) {
                        currNeighbor.at(l) = 0;
                    }
                }

                double distance = euclideanDistance(classifyObject, currNeighbor);

                //Finds nearest neighbor and correctly classifies it
                if (distance < nearestNeighborDist) {
                    nearestNeighborDist = distance;
                    nearestNeighborLoc = j;
                    labelOfNearestNeighbor = classifyInstance(data[j]);
                }

            }
        }
        //Counts how many instances we correctly classified
        if (currInstanceLabel == labelOfNearestNeighbor) {
            ++correctlyClassifiedCnt;
        }
    }
    //Return k fold cross validation
    float accuracy = (correctlyClassifiedCnt / data.size());
    return accuracy;
}

//Forward Selection Algorithm
void forwardSelection(vector<vector <double> > data, int numFeatures) {
    //Holds features already evaluated and remove from next level of search tree
    vector <int> currFeatures;

    //Get accuracy with no features and print it to screen
    float overallBestAcc = leaveOneOutCrossValidation(data, numFeatures, currFeatures, -1);
    cout << "Running nearest neighbor with no features, using \"leave-one-out\" evaluation, I get an accuracy of " << overallBestAcc * 100 << "%" << endl;

    cout << "Beginning search." << endl;

    //Used for formatting purposes
    bool firstRound = true;

    //Needed to correctly return subset with highest accuracy start it with default rate subset, so no feature accuracy
    vector <int> bestFeatureSubset = currFeatures;
    vector<vector<int> > allFeatureSubsets;

    // 2. For loop that walks down the tree for all features (outer for loop)
    for (int i = 0; i < numFeatures; ++i) {        
        //Holds feature we added at each level of the tree
        int featureToAdd;

        //The best accuracy for current level of tree
        float bestAccuracy = 0;

        // 3. For loop that considers each feature (inner for loop) and skips the first as that is the class label
        for (int j = 1; j < numFeatures + 1; ++j) {
            //Check if current feature has not already been chosen
            bool newFeature = true;
            for (int k = 0; k < currFeatures.size(); ++k) {
                if (currFeatures.at(k) == j) {
                    newFeature = false;
                    break;
                }
            }
            
            //If it is not a new feature move to next feature
            if (!newFeature) {
                continue;
            }

            // Cross validation test inside this loop, pass in all necessary data
            float accuracy = leaveOneOutCrossValidation(data, numFeatures, currFeatures, j);

            //Outputs each feature's accuracy along with subsets of features
            if (firstRound) {
                //Prints out first feature subset correctly
                cout << "\tUsing feature(s) {" << j << "} accuracy is " << accuracy * 100 << "%" << endl;
            }

            //Prints out subset of features correctly
            else {
                cout << "\tUsing feature(s) {";
                for (int r = 0; r < currFeatures.size(); ++r) {
                    if (currFeatures.size() == 1) {
                        cout << currFeatures.at(r) << ", ";
                    }

                    else if (r != currFeatures.size()) {
                        cout << currFeatures.at(r) << ", ";
                    }
                }
                cout << j << "} accuracy is " << accuracy * 100 << "%" << endl;
            }

            //Store best accuracy so far and add that feature to currFeatures
            // If current feature has a higher accuracy so far we replace it and add it as the feature we need for that level of the tree
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                featureToAdd = j;
                //Add the current subset of features that resulted in the highest accuracy at this level of the tree
                allFeatureSubsets.push_back(currFeatures);
            }
        }

        //Stores overall highest accuracy and pushes back the subset that resulted in that highest accuracy
        if (bestAccuracy > overallBestAcc) {
            overallBestAcc = bestAccuracy;
            bestFeatureSubset.push_back(featureToAdd);
        } 

        //After first level of tree, formatting not needed
        firstRound = false;

        //Adds feature if it has a higher accuracy
        currFeatures.push_back(featureToAdd);

        //Correctly output subset of features that result in the highest current accuracy
        cout << "Feature set {";
        for (int m = 0; m < currFeatures.size(); ++m) {
            if (m != currFeatures.size() - 1) {
                cout << currFeatures.at(m) << ", ";
            }

            else {
                cout << currFeatures.at(m) << "} was best, accuracy is " << bestAccuracy * 100 << "%" << endl;
            }
        }
    }

    //After going through the tree levels, print out the best subset along with its accuracy which is the highest at this point
    cout << "Finished search!! The best feature subset is {";
    for (int n = 0; n < bestFeatureSubset.size(); ++n) {
        if (n != bestFeatureSubset.size() - 1) {
            cout << bestFeatureSubset.at(n) << ", ";
        }

        else {
            cout << bestFeatureSubset.at(n) << "}, which has an accuracy of " << overallBestAcc * 100 << "%" << endl;
        }
    }
}

//Takes in vector of all the data from file and turns it into its respective row
vector<vector<double> > turnDataIntoRows(vector<double> fullData, int numFeatures) {
    //Converts data into a vector of vector to store each instance/row
    vector<vector<double> > instances;

    //Needed to end while loop
    int numRows = fullData.size() / (numFeatures + 1);
    
    //i will hold current row and we start at the first row
    int i = 0;

    //Holds next row data
    int nextRow = 0;

    while (i < numRows) {
        //create a vector for the current row
        vector <double> row;

        //Add all features for current row to row vector
        for (int j = nextRow; j < (numFeatures + 1) + nextRow; j++) {
            row.push_back(fullData.at(j));
        }

        //Push into each row/instance into the vector of vectors
        instances.push_back(row);

        //Move to next row
        nextRow += (numFeatures + 1);
        
        //Increment i to go to the next row
        i++;
    }
    
    return instances;
}

int main() {
    //Follows fornat for traceback
    cout << "Welcome to the David Aispuro Feature Selection Algorithm." << endl;

    //Allows user to enter in file
    string fileName;

    cout << "Type in the name of the file to test : ";
    getline(cin, fileName);

    //Holds the contents of the file
    string fileContents;

    //Opens the file the user entered
    // 1. Load file in for input to algorithm
    ifstream file(fileName);

    //Stores number of features
    int numberOfFeatures = 0;

    //Used to only read first line
    bool isFirstLine = true;

    //Can handle IEEE numbers
    vector<double> data;

    //Open file and group features into their columns
    if (file.is_open()) {
        while (getline(file, fileContents)) {
            //Read in line by line
            stringstream line(fileContents);
            //Read in each feature
            double feature;

            //Reads in each feature stores each value
            while (line >> feature) {
                data.push_back(feature);
            }

            //Uses the first line to determine number of features
            if (isFirstLine) {
                numberOfFeatures = data.size() - 1;
                isFirstLine = false;
            }
        }
    }

    file.close();

    //Gets all the features arranged in their respective rows
    vector<vector<double> > allRows = turnDataIntoRows(data, numberOfFeatures);

    cout << endl;

    //Follows format for traceback which shows number of features and instances of data
    cout << "The dataset has " << numberOfFeatures << " features (not including the class attribute), with " << allRows.size() << " instances." << endl;

    forwardSelection(allRows, numberOfFeatures);

    return 0;
}