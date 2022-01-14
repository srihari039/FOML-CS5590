# FoML Assign 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

# importing numpy to manipulate data
import numpy as np
# importing csv to retrieve data from CSV files
import csv

# Enter You Name Here
myname = "Sri-Hari " # #

# Decision node
# Contains Index of split, threshold, information gain, left sub tree, right sub tree
class decisionNode():
    def __init__ (self, splitObj, leftNode = None, rightNode = None):
        self.fIndex = splitObj.get("fIndex")
        self.threshold = splitObj.get("threshold")
        self.informationGain = splitObj.get("informationGain")
        self.left = leftNode
        self.right = rightNode
        self.isLeaf = False

# leaf Node
# Contains only value as it is the leaf
class leafNode():
    def __init__ (self,value = None):
        self.value = value
        self.isLeaf = True

# Implement your decision tree below
# Decision tree
class DecisionTree():
      
    # setting the depth essentially(improvement)
    shallowDepth = None
    # tree
    tree = {}

    # Build tree function
    def buildTree(self, training_set, curDepth = 0):
        
        # Abstract the features of the training set
        features = training_set[:,:-1]
        # Abrstract the results of the features
        result = training_set[:,-1]
        
        # n(number of features)
        numberOfFeatures = len(features[0])
        
        # Adding the check to stop the overfitting of the tree
        if curDepth <= self.shallowDepth:
        
            # get the best split
            bestSplit = self.getBestSplit(training_set,numberOfFeatures)

            # If information Gain of the obtained best split is positive,
            # build the left sub tree, right sub tree and return the decision node
            # decision node contains, index of classification, threshold value, left tree, right tree, information gain
            if bestSplit.get("informationGain"):
                if bestSplit["informationGain"] > 0:
                    leftSubtree = self.buildTree(bestSplit["leftData"],curDepth+1)
                    rightSubtree = self.buildTree(bestSplit["rightData"],curDepth+1)
                    return decisionNode(bestSplit,leftSubtree,rightSubtree)
                    
        # We keep a leaf in the tree, Either if depth is reached to max depth or else when the
        # information gain is 0, then it can be classified easily(only one type of class is present)
        
        # Returns the key which is repeated max times in the list
        resultList = list(result)
        return leafNode(max(resultList, key=resultList.count))

    # utility function to store the attributes and return a python dictionary
    def fitInsplit(self,split={},index=None,threshold=None,gain=None):
        split["fIndex"] = index
        split["threshold"] = threshold
        split["informationGain"] = gain
        return split

    # function to get best split
    def getBestSplit(self, training_set, numberOfFeatures):
        
        # make an empty dictionary
        bestSplit = {}
        # mark initial gain is negative infinity
        infoGain = -float("Inf")
        
        # Iterate over the index over the number of features
        for index in range(numberOfFeatures):
            
            value = training_set[:,index]
            possibleThresholds = np.unique(value)

            # Iterate over the possible thresholds
            for threshold in possibleThresholds:
            
                # split the data into two, left and right
                leftData = []
                rightData = []
                for row in training_set:
                    (rightData,leftData)[row[index] <= threshold].append(row)
                leftData = np.asarray(leftData)
                rightData = np.asarray(rightData)

                # Compute the information Gain and update the best splits
                if len(leftData) and len(rightData):
                    y = training_set[:,-1]
                    leftY = leftData[:,-1]
                    rightY = rightData[:,-1]

                    # Computing the information gain
                    # presentGain = self.getGiniInformationGain(y,leftY,rightY)
                    presentGain = self.getEntropyInformationGain(y,leftY,rightY)

                    if presentGain > infoGain:
                        bestSplit = self.fitInsplit(bestSplit,index,threshold,infoGain)
                        bestSplit["leftData"] = leftData
                        bestSplit["rightData"] = rightData
                        infoGain = presentGain

        return bestSplit

    # Gini Index
    def getGiniIndex(self, y):
        # Initialize gini index with 1
        gini = 1
        
        # calculate the probability of each class present and 
        # substract the square of it from gini 
        # It essentially gives the value between 0 and 0.5
        if '0' in y:
            prob = np.count_nonzero(y == '0') /len(y)
            gini -= prob ** 2
        if '1' in y:
            prob = np.count_nonzero(y == '1') /len(y)
            gini -= prob ** 2

        # return gini index
        return gini
    
    # Entropy
    def getEntropy(self,y):
        # Initialize entropy with 0
        entropy = 0
        
        # calculate the probability of each class present and
        # add -prob*(log2(prob)) to the entropy
        # It essentially gives the value between 0 and 1
        if '0' in y:
            prob = np.count_nonzero(y == '0') /len(y)
            entropy += -prob*np.log2(prob)
        if '1' in y:
            prob = np.count_nonzero(y == '1') /len(y)
            entropy += -prob*np.log2(prob)
        
        # return entropy
        return entropy
    
    # Calculating the information gain from Gini Index
    def getGiniInformationGain(self, parent, lChild, rChild):
        
        # mark the length of the parent
        lenParent = len(parent)
        
        # calculate the gini index of parent
        gain = self.getGiniIndex(parent)
        
        # substract the weighted gini index from the gain obtained from parent
        gain -= (len(lChild)* self.getGiniIndex(lChild))/lenParent
        gain -= (len(rChild)* self.getGiniIndex(rChild))/lenParent
        
        # return the information gain
        return gain
    
    # Calculating the information gain from Entropy    
    def getEntropyInformationGain(self,parent,lchild,rchild):

        # mark the length of the parent
        lenParent = len(parent)
        
        # calculate the entropy of parent
        gain = self.getEntropy(parent)
        
        # substract the weighted entropy from the gain obtained from parent        
        gain -= (len(lchild)* self.getEntropy(lchild))/lenParent
        gain -= (len(rchild)* self.getEntropy(rchild))/lenParent
        
        # return the information gain
        return gain

    # implement this function
    # Function which builds tree
    def learn(self, training_set):
        
        # converting it to numpy array for further manipulation
        training_data = np.array(training_set)
        # noting the shape of data
        size,depth = training_data.shape
        # assigning the max depth
        self.shallowDepth = depth/4
        # Build the tree and store it the tree
        self.tree = self.buildTree(training_data)
        
    # implement this function
    # Function which classifies based on the features and the existing tree
    def classify(self,test_instance,dtree=None):
    
        # if it is a valid decision node
        if dtree:
            # if value is a truthy value, return the value
            # Decision node has the value attribute as falsy and leaf node has truthy value
            if dtree.isLeaf:
                # return leaf value
                return dtree.value
            
            # collect the value of test instance at the row, according to the data present in decision node
            value = test_instance[dtree.fIndex]

            # based on the value, return the answer from left and right sub tree
            return self.classify(test_instance,dtree.left) if value <= dtree.threshold \
                else self.classify(test_instance,dtree.right)
        
        # Just a tweak, not to write other function and not to change the definition of
        # classify method given in the resources
        else:
            dtree = self.tree
            return self.classify(test_instance,dtree)
        

def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print ("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.

    # global accuracy set to 0
    accuracy = 0
    # Working on 10-cross fold validation
    K = 10
    # loop over the remainders from 0 to K
    for rem in range(0,K):
        
        # Add the row to training set if index doesn't leave remainder rem
        # Add the row to testing set if index leaves remainder rem
        training_set = [x for i, x in enumerate(data) if i % K != rem]
        test_set = [x for i, x in enumerate(data) if i % K == rem]       

        # create a tree instance
        tree = DecisionTree()
        # Construct a tree using training set
        # training_set = training_set[:100]
        tree.learn(training_set)

        # Classify the test set using the tree we just constructed
        results = []
        for instance in test_set:
            result = tree.classify(instance[:-1])
            results.append(result == instance[-1])

        # Accuracy
        kaccuracy = float(results.count(True))/float(len(results))
        print(rem,"completed with accuracy ",kaccuracy)
        accuracy += kaccuracy

    accuracy /= K
    print("accuracy: %.4f" % accuracy)
    
    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()

if __name__ == "__main__":
    run_decision_tree()
