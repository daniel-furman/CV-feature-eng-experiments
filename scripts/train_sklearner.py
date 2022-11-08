
"""
Script to train a shallow learner on the chesapeake bay dataset
"""
# modeling libraries
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def scikit_optuna_pipeline(train_images, train_labels, val_images, val_labels):

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

        return balanced_accuracy_score(val_labels, val_predictions)

    study = optuna.create_study(direction='maximize')
    study.optimize(balanced_acc_objective, n_trials=25)

    trial = study.best_trial

    print('Balanced Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))