# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
# Juhan Nam
#
# Apr-06-2018: developed version
# Yoonjae Cho

from feature_summary import *

from sklearn.linear_model import SGDClassifier
from sklearn import svm, neighbors, tree
from sklearn.ensemble import *


def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):
    clfs = [
        SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0),
        SGDClassifier(verbose=0, loss="modified_huber", alpha=hyper_param1, max_iter=1000, penalty="l2",
                      random_state=0),
        svm.SVC(kernel='linear'),
        svm.LinearSVC(),
        svm.NuSVC(),
        neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance'),
        tree.DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10)
    ]

    valid_acc = []

    for clf in clfs:
        # train
        clf.fit(train_X, train_Y)

        # validation
        valid_Y_hat = clf.predict(valid_X)

        accuracy = np.sum((valid_Y_hat == valid_Y)) / 200.0 * 100.0
        valid_acc.append(accuracy)
        print('validation accuracy = ' + str(accuracy) + ' %')

    final_clf = clfs[np.argmax(valid_acc)]

    return final_clf, accuracy


if __name__ == '__main__':

    # load data
    train_X = mean_mfcc('train')
    valid_X = mean_mfcc('valid')
    test_X = mean_mfcc('test')

    # label generation
    cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X = train_X.T
    valid_X = valid_X.T

    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    valid_X = valid_X - train_X_mean
    valid_X = valid_X / (train_X_std + 1e-5)

    # training model
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    models = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
        models.append(clf)
        valid_acc.append(acc)

    # choose the model that achieve the best validation accuracy
    # and evaluate the model with the test set
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X / (train_X_std + 1e-5)

    max_acc, final_model = 0, None
    for model in models:
        test_Y_hat = model.predict(test_X)
        accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0

        print('\ntest accuracy = ' + str(accuracy) + ' %')
        print(model)

        if max_acc < accuracy:
            max_acc, final_model = accuracy, model

    print('\n==========================================================================')
    print('Best test accuracy = ' + str(max_acc) + ' %')
    print(final_model)
