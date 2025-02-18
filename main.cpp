#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
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

    //Iterate through each feature and calculate the sum
    for (int i = 0; i < currentObj.size(); ++i) {
        sum += pow((currentObj.at(i) - neighbor.at(i)), 2);
    }

    //Returns square distance/euclidean distance
    return sqrt(sum);
}

//Accuracy function needed for forward selection
float leaveOneOutCrossValidation(vector<vector <double> >& data) {
    //Count need to check how many instances we classified correctly
    float correctlyClassifiedCnt = 0;

    //Iterates through each row
    for (int i = 0; i < data.size(); ++i) {
        //We want to classify each object based on its features
        vector<double> classifyObject = data.at(i);

        //We erase the first data point as it is simply the class label
        classifyObject.erase(classifyObject.begin());

        //Returns the label of the current instance
        int currInstanceLabel = classifyInstance(data[i]);

        //Initially nearest neighbor is infinity away
        double nearestNeighborDist = 1000000;
        double nearestNeighborLoc = 1000000;

        //Need the class of the nearest neighbor to see if we classifies it correctly
        int labelOfNearestNeighbor;

        //Finds nearest neighbor by comparing to each other instance
        for (int j = 0; j < data.size(); ++j) {
            //Current instance does not check itself
            if (i != j) {
                //Neighbor we want to compare to
                vector<double> currNeighbor = data.at(j);

                //We erase the first data point as it is simply the class label
                currNeighbor.erase(currNeighbor.begin());

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
//Currently not using features but runs how it shows in the video for testing purposes
void forwardSelection(vector<vector <double> > features, int numFeatures) {
    //Holds features already evaluated and remove from next level of search tree
    vector <int> currFeatures;

    //Needed for rand, temporary
    srand(time(0));

    // 2. For loop that walks down the tree for all features (outer for loop)
    for (int i = 0; i < numFeatures; ++i) {
        cout << "On the "<< i + 1 << "th level of the search tree" << endl;
        
        //Hold the best accuracy we have encountered
        int bestAccuracy = 0;

        //Holds feature we added at each level of the tree
        int featureToAdd;

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

            //Testing stub needed for which generates a number between 1-100
            // Cross validation test inside this loop, but to begin just use random()
            int accuracy = rand() % 100 + 1;

            //Still testing wanted to make sure it was correctly outputting accuracy
            cout << "Using feature " << j << " accuracy is " << accuracy << "%" << endl;

            //Store best accuracy so far and add that feature to currFeatures
            // If current feature has a higher accuracy so far we replace it and add it as the feature we need for that level of the tree
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                featureToAdd = j;
            }
        }

        //Adds feature if it has a higher accuracy
        currFeatures.push_back(featureToAdd);
        //Not done yet just used to confirm highest percentage is being added
        cout << "Feature set " << featureToAdd << " was best, accuracy is " << bestAccuracy << "%" << endl;
        cout << endl;
    }
}

//Takes in vector of all the data from file and turns it into its respective row
vector<vector<double> > turnFeaturesIntoRows(vector<double> fullData, int numFeatures) {
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
    //Allows user to enter in file
    string fileName;

    cout << "Enter the name of the file: ";
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
    vector<vector<double> > allRows = turnFeaturesIntoRows(data, numberOfFeatures);

    forwardSelection(allRows, numberOfFeatures);

    cout << leaveOneOutCrossValidation(allRows) << endl;

    return 0;
}