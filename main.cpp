#include <iostream>
using namespace std;

// Forward Selection as explained in the video
// 1. Load file in for input to algorithm
    // Side note: use a utility function for standard IEEE numbers in file 
// 2. For loop that walks down the tree for all features (outer for loop)
    // Side note: Remember first column is class label so data - 1 for real size
    // "On the ith level of the search tree"
// 3. For loop that considers each feature (inner for loop)
    // "Considering adding the i feature"
    // Cross validation test inside this loop, but to begin just use random()
    // If current feature has a higher accuracy so far we replace it and add it as the feature we need for that level of the tree
    // Also need an array that stores features we already looked at and remove from next level of search tree

int main() {

    return 0;
}