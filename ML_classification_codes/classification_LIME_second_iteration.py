import sys
import numpy as np
import csv
import joblib
import lime
from lime.lime_tabular import LimeTabularExplainer
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from xgboost import XGBClassifier

def load_data(file_path):
    threshold = 0.3
    data = np.loadtxt(file_path)
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]

    class_counts = Counter(y)
    imbalance_ratio = {cls: count / len(y) for cls, count in class_counts.items()}
    classes_with_larger_samples = [cls for cls, ratio in imbalance_ratio.items() if ratio > threshold]

    if classes_with_larger_samples:
        print("Samples kept and removed for each class:")
        desired_counts = {cls: int(count * (1 - threshold)) for cls, count in class_counts.items()}

        for cls in classes_with_larger_samples:
            class_indices = np.where(y == cls)[0]

            num_samples_to_remove = class_counts[cls] - desired_counts[cls]

            indices_to_keep = np.random.choice(class_indices, desired_counts[cls], replace=False)
            print(f"Class {cls}:")
            print(f"\tSamples kept: {desired_counts[cls]}")
            print(f"\tSamples removed: {num_samples_to_remove}")

            X = np.delete(X, indices_to_keep, axis=0)
            y = np.delete(y, indices_to_keep)

    return X, y

def extra_trees_classifier(X_train, y_train):
    clf = ExtraTreesClassifier(n_estimators=25, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def mlp_classifier(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def svm_classifier(X_train, y_train):
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def xgboost_classifier(X_train, y_train):
    clf = XGBClassifier(learning_rate=0.3, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None
    return accuracy, precision, recall, f1, logloss

def write_model_metrics_to_csv(file_path, accuracy, precision, recall, f1, logloss):
    with open(file_path, mode='w', newline='') as eval_file:
        eval_writer = csv.writer(eval_file)
        eval_writer.writerow(['Metric', 'Value'])
        eval_writer.writerow(['Accuracy', accuracy])
        eval_writer.writerow(['Precision', precision])
        eval_writer.writerow(['Recall', recall])
        eval_writer.writerow(['F1 Score', f1])
        if logloss is not None:
            eval_writer.writerow(['Log Loss', logloss])

def calculate_feature_importance(clf, X_train, y_train, num_samples_per_class='all'):
    if isinstance(clf, (MLPClassifier, ExtraTreesClassifier, XGBClassifier)):
        classes = np.unique(y_train)
        num_classes = len(classes)
        print(f"Number of classes: {num_classes}")

        num_samples_per_class = 100

        explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode='classification', random_state=42)
        importances = []

        clnum = 1
        for cls in classes:
            class_indices = np.where(y_train == cls)[0]
            
            if num_samples_per_class == 'all' or num_samples_per_class >= len(class_indices):
                selected_indices = class_indices
            else:
                selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)

            class_samples = X_train[selected_indices]
            
            class_importance = np.zeros(X_train.shape[1])
           
            samplenum = 1
            for sample in class_samples:

                print('class =', clnum, 'num_samples =',samplenum)

                exp = explainer.explain_instance(sample, clf.predict_proba, num_features=X_train.shape[1])
                
                importance = np.zeros(X_train.shape[1])
                for i in exp.as_list():
                    if len(i[0].split()) == 3:
                        feature_idx = int(i[0].split()[0])
                        importance[feature_idx] = i[1]
                    else:
                        feature_idx = int(i[0].split()[2])
                        importance[feature_idx] = i[1]
                
                class_importance += importance
                samplenum+=1
            
            class_importance /= len(class_samples)
            importances.append(class_importance)
            clnum+=1
        
        average_importance = np.mean(importances, axis=0)
        return average_importance
    else:
        if hasattr(clf, 'feature_importances_'):
            return clf.feature_importances_
        else:
            print("Feature importances are not available for the given classifier.")
            return None

def write_feature_importance_to_txt(file_path, feature_importance):
    with open(file_path, mode='w') as importance_file:
        for idx, importance in enumerate(feature_importance):
            importance_file.write(f"Feature_{idx} {importance}\n")

def save_fold_data(X, y, fold_num, fold_type):
    data = np.column_stack((X, y))
    np.savetxt(f"{fold_type}_fold_{fold_num}.txt", data, delimiter=' ', fmt='%s')

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    X, y = load_data(data_file)

    classifiers = [
        ('MLP', mlp_classifier),
        ('Extra_Trees', extra_trees_classifier),
        ('XGBoost', xgboost_classifier),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for clf_name, clf_func in classifiers:
        print(f"\n{clf_name.replace('_', ' ')} 5-fold Cross-Validation:")

        accuracy_list, precision_list, recall_list, f1_list, logloss_list = [], [], [], [], []

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            save_fold_data(X_test, y_test, fold + 1, 'test')

            classifier = clf_func(X_train, y_train)
            accuracy, precision, recall, f1, logloss = evaluate_classifier(classifier, X_test, y_test)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            logloss_list.append(logloss)

            logloss_str = f"{logloss:.4f}" if logloss is not None else "N/A"
            print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Log Loss: {logloss_str}")

            fold_file_path = f'{clf_name}_fold_{fold+1}_metrics.csv'
            write_model_metrics_to_csv(fold_file_path, accuracy, precision, recall, f1, logloss)

            feature_importance = calculate_feature_importance(classifier, X_train, y_train)
            if feature_importance is not None:
                feature_importance_file_path = f'{clf_name}_fold_{fold+1}_feature_importance.txt'
                write_feature_importance_to_txt(feature_importance_file_path, feature_importance)
                print(f"LIME feature importance for {clf_name.replace('_', ' ')} saved to: {feature_importance_file_path}")

            if clf_name == 'Extra_Trees':
                inherent_importance = classifier.feature_importances_
                inherent_importance_file_path = f'{clf_name}_fold_{fold+1}_inh_FI_feature_importance.txt'
                write_feature_importance_to_txt(inherent_importance_file_path, inherent_importance)
                print(f"Inherent feature importance for {clf_name.replace('_', ' ')} saved to: {inherent_importance_file_path}")

            model_file_path = f'{clf_name}_fold_{fold+1}_model.joblib'
            joblib.dump(classifier, model_file_path)
            print(f"Model for fold {fold+1} saved to: {model_file_path}")

        mean_accuracy = np.mean(accuracy_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_f1 = np.mean(f1_list)
        mean_logloss = np.mean([logloss for logloss in logloss_list if logloss is not None])
        logloss_str = f"{mean_logloss:.4f}" if mean_logloss is not None else "N/A"
        print(f"\nMean Metrics over 5 folds - Accuracy: {mean_accuracy:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, F1-score: {mean_f1:.4f}, Log Loss: {logloss_str}")

if __name__ == "__main__":
    main()
