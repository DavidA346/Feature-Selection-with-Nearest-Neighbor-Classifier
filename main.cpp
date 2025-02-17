#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

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

        // 3. For loop that considers each feature (inner for loop)
        for (int j = 0; j < numFeatures; ++j) {
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

            cout << "Considering adding the " << j + 1 << " feature" << endl;

            //Testing stub needed for which generates a number between 1-100
            // Cross validation test inside this loop, but to begin just use random()
            int accuracy = rand() % 100 + 1;

            //Store best accuracy so far and add that feature to currFeatures
            // If current feature has a higher accuracy so far we replace it and add it as the feature we need for that level of the tree
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                featureToAdd = j;
            }
        }
        //Adds feature if it has a higher accuracy
        currFeatures.push_back(featureToAdd);
    }
}

//Takes in vector of all the data from file and turns it into columns for each feature
vector<vector<double> > turnFeaturesIntoColumns(vector<double> fullData, int numFeatures) {
    //Converts data into a vector of vector to store the columns
    vector<vector<double> > features;
    
    //i will hold current column and we start at the first column
    int i = 0;
    while (i < numFeatures) {
        //create a vector for the current column
        vector <double> column;

        //Add values for current feature by incrementing by number of features in order to get to next now of data
        for (int j = i; j < fullData.size(); j += numFeatures) {
            column.push_back(fullData.at(j));
        }

        //Push into features double vector
        features.push_back(column);
        
        //Increment i to go to next feature
        ++i;
    }
    
    return features;
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

            //Skip first column as it is just the classes and not an actual feature
            line >> feature;

            //Reads in each feature stores each value
            while (line >> feature) {
                data.push_back(feature);
            }

            //Uses the first line to determine number of features
            if (isFirstLine) {
                numberOfFeatures = data.size();
                isFirstLine = false;
            }
        }
    }

    file.close();

    //Gets all the features arranged in their respective columns
    vector<vector<double> > featuresInColumns = turnFeaturesIntoColumns(data, numberOfFeatures);

    forwardSelection(featuresInColumns, numberOfFeatures);

    return 0;
}

