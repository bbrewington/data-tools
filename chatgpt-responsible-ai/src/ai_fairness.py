from aif360.metrics import ClassificationMetric
from sklearn.metrics import confusion_matrix

def monitor_for_bias(model, test_data, privileged_group, unprivileged_group):
    # Make predictions on the test data using the model
    y_pred = model.predict(test_data.drop('churn', axis=1))

    # Calculate the confusion matrix and classification report
    tn, fp, fn, tp = confusion_matrix(test_data['churn'], y_pred).ravel()

    # Calculate the equal opportunity difference using the AIF360 package
    cm = ClassificationMetric(test_data['churn'], y_pred, 
                              privileged_groups=[{privileged_group: 1}], 
                              unprivileged_groups=[{unprivileged_group: 1}])
    eod = cm.equal_opportunity_difference()

    return {'confusion_matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
            'classification_report': classification_report(test_data['churn'], y_pred),
            'equal_opportunity_difference': eod}
    
def choose_features(df, excluded_features):
    # Define a list of all features in the dataset
    all_features = df.columns.tolist()

    # Remove the excluded features from the list
    features_to_include = [f for f in all_features if f not in excluded_features]

    return features_to_include