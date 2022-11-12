
"""
Script to train a shallow learner on the chesapeake bay dataset
"""

# modeling libraries
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


def scikit_optuna_pipeline(train_images, train_labels, val_images, val_labels, n_trials):
    
    def balanced_acc_objective(trial):
        
        classifier = trial.suggest_categorical('classifier', ['LR', 'FFNN'])
    
        if classifier == 'LR':
            max_iter=trial.suggest_int('max_iter', 500, 1500)
            solver = trial.suggest_categorical("solver", ['newton-cg', 'sag', 'saga', 'lbfgs'])
              
            clf = LogisticRegression(
                max_iter=max_iter, solver=solver)
        else:
            n = trial.suggest_int('hidden_layer_sizes', 32, 256)
            activation = trial.suggest_categorical("activation", ['relu', "tanh", "logistic"])

            clf = MLPClassifier(hidden_layer_sizes=(n,),
                            activation=activation,
                            learning_rate='adaptive')
        
        clf.fit(train_images, train_labels)
        val_predictions = clf.predict(val_images)

        bac = balanced_accuracy_score(val_labels, val_predictions)

        return bac

    study = optuna.create_study(direction='maximize')
    study.optimize(balanced_acc_objective, n_trials=n_trials)

    trial = study.best_trial

    print('Balanced Accuracy Validation Set Optuna: {}'.format(trial.value))
    print("Best hyperparameters Optuna: {}".format(trial.params))

    if trial.params['classifier'] == 'FFNN':
        n = trial.params['hidden_layer_sizes']
        act = trial.params['activation']
        clf = MLPClassifier(hidden_layer_sizes=(n,),
                            activation=act,
                            learning_rate='adaptive')
        clf.fit(train_images, train_labels)
        val_predictions = clf.predict(val_images)
        bac = balanced_accuracy_score(val_labels, val_predictions)
        print('Balanced Accuracy Validation Set Retrained Model: {}'.format(bac))

    else:
        max_iter = trial.params['max_iter']
        solver = trial.params['solver']
        clf = LogisticRegression(max_iter=max_iter, solver=solver)
        clf.fit(train_images, train_labels)
        val_predictions = clf.predict(val_images)
        bac = balanced_accuracy_score(val_labels, val_predictions)
        print('Balanced Accuracy Validation Set Retrained Model: {}'.format(bac))

    return clf



def scikit_optuna_pipeline_hog(train_images, train_labels, val_images, val_labels, n_trials):
    
    def balanced_acc_objective(trial):
        
        classifier = trial.suggest_categorical('classifier', ['RF', 'FFNN'])
    
        if classifier == 'RF':
            n_estimators=trial.suggest_int('n_estimators', 32, 70)              
            clf = RandomForestClassifier(
                n_estimators=n_estimators)
        else:
            n = trial.suggest_int('hidden_layer_sizes', 32, 70)
            activation = trial.suggest_categorical("activation", ['relu', "tanh", "logistic"])

            clf = MLPClassifier(hidden_layer_sizes=(n,),
                            activation=activation,
                            learning_rate='adaptive')
        
        clf.fit(train_images, train_labels)
        val_predictions = clf.predict(val_images)

        bac = balanced_accuracy_score(val_labels, val_predictions)

        return bac

    study = optuna.create_study(direction='maximize')
    study.optimize(balanced_acc_objective, n_trials=n_trials)

    trial = study.best_trial

    print('Balanced Accuracy Validation Set Optuna: {}'.format(trial.value))
    print("Best hyperparameters Optuna: {}".format(trial.params))

    if trial.params['classifier'] == 'FFNN':
        n = trial.params['hidden_layer_sizes']
        act = trial.params['activation']
        clf = MLPClassifier(hidden_layer_sizes=(n,),
                            activation=act,
                            learning_rate='adaptive')
        clf.fit(train_images, train_labels)
        val_predictions = clf.predict(val_images)
        bac = balanced_accuracy_score(val_labels, val_predictions)
        print('Balanced Accuracy Validation Set Retrained Model: {}'.format(bac))

    else:
        max_iter = trial.params['max_iter']
        solver = trial.params['solver']
        clf = LogisticRegression(max_iter=max_iter, solver=solver)
        clf.fit(train_images, train_labels)
        val_predictions = clf.predict(val_images)
        bac = balanced_accuracy_score(val_labels, val_predictions)
        print('Balanced Accuracy Validation Set Retrained Model: {}'.format(bac))

    return clf