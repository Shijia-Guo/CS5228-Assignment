import numpy as np
import os
import json
import operator
import sys
def best_split(X,y):
    split_error = sys.float_info.max
    splitting_variable = 0
    splitting_threshold = 0.0
    splitting_left_indexes = []
    splitting_right_indexes = []
    for feature in range(0,X.shape[1]):
        # construct the split value, the split index is feature
        split_index_values = []
        for index in range(0,X.shape[0]):
            split_index_values.append(X[index][feature])
        split_index_values.sort()
        split_index_threadhold_values = split_index_values[0:len(split_index_values)-1]

        for threadhold in split_index_threadhold_values:
            [left_split_indexes,right_split_indexes] = check_split(X,feature,threadhold)
            loss = calculate_split_error(y,left_split_indexes,right_split_indexes)
            if loss < split_error:
                split_error = loss
                splitting_variable = feature
                splitting_threshold = threadhold
                splitting_left_indexes = left_split_indexes
                splitting_right_indexes = right_split_indexes
    left_x = np.array([X[i].tolist() for i in splitting_left_indexes ])
    left_y = np.array([y[i] for i in splitting_left_indexes ])
    right_x = np.array([X[i].tolist() for i in splitting_right_indexes ])
    right_y = np.array([y[i] for i in splitting_right_indexes ])
    return {'splitting_variable':splitting_variable,'splitting_threshold':splitting_threshold,'left_x':left_x,'left_y':left_y,'right_x':right_x,'right_y':right_y}

def split_node(root,height,node,max_height,min_samples_split):
    left_x = node['left_x']
    left_y = node['left_y']
    right_x = node['right_x']
    right_y = node['right_y']
    root['splitting_variable'] = node['splitting_variable']
    root['splitting_threshold'] = node['splitting_threshold']
    if height >= max_height:
        root['left'] = np.average(left_y)
        root['right'] = np.average(right_y)
        return
    
    if(len(left_y)<min_samples_split):
         root['left'] = np.average(left_y)
    else:
        left_node = best_split(left_x,left_y)
        root['left'] = dict()
        split_node(root['left'],height+1,left_node,max_height,min_samples_split)

    if(len(right_y)<min_samples_split):
         root['right'] = np.average(right_y)
    else:
        right_node = best_split(right_x,right_y)
        root['right'] = dict()
        split_node(root['right'],height+1,right_node,max_height,min_samples_split)


def calculate_split_error(y,left_split_indexes,right_split_indexes):
    y_left_avg = 0.0
    y_right_avg = 0.0
    loss = 0.0
    for i in left_split_indexes:
        y_left_avg = y_left_avg + y[i]
    y_left_avg = y_left_avg/len(left_split_indexes)

    for j in right_split_indexes:
        y_right_avg = y_right_avg + y[j]
    y_right_avg = y_right_avg/len(right_split_indexes)

    for k in range(len(y)):
        if k in left_split_indexes:
            loss = loss + (y[k] - y_left_avg) **2
        else:
            loss = loss + (y[k] - y_right_avg) **2
    return loss

def check_split(X,feature,threadhold):
    left_split_indexes = []
    right_split_indexes =[]
    for i in range(X.shape[0]):
        if(X[i][feature] <= threadhold):
            left_split_indexes.append(i)
        else:
            right_split_indexes.append(i)
    return [left_split_indexes,right_split_indexes]
def predict_single_value(X,node):
    if(X[node['splitting_variable']] <= node['splitting_threshold']):
        if isinstance(node['left'],dict):
            return predict_single_value(X,node['left'])
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_single_value(X,node['right'])
        else:
            return node['right']

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = dict()
    def fit(self, X, y):
        node = best_split(X,y)
        split_node(self.root,1,node,self.max_depth,self.min_samples_split)
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        predict_values=[]
        for i in range(X.shape[0]):
            predict_values.append(predict_single_value(X[i],self.root))
        return np.array(predict_values)

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test

if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)
            
            model_string = tree.get_model_string()
            
            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)
            print(operator.eq(model_string, test_model_string))
            y_pred = tree.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
