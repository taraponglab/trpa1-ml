import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, roc_curve, matthews_corrcoef, precision_score, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold, LeaveOneOut
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

def remove_constant_string_des(df):
    #delete string value
    df = df.select_dtypes(exclude=['object'])
    #delete constant value
    for column in df.columns:
        if df[column].nunique() == 1:  # This checks if the column has only one unique value
            df = df.drop(column, axis=1)  # This drops the column from the DataFrame
    return df

def remove_highly_correlated_features(df, threshold=0.7):
    # Compute pairwise correlation of columns
    corr_matrix = df.corr().abs()
    # Create a mask for the upper triangle
    upper = corr_matrix.where(
        pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), 
                     index=corr_matrix.index, columns=corr_matrix.columns)
    )
    # Identify columns to drop based on threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop the columns from the DataFrame
    df_dropped = df.drop(columns=to_drop)
    return df_dropped

def y_prediction(model, x_train, y_train, col_name):
    y_pred = pd.DataFrame(model.predict(x_train), columns=[col_name]).set_index(y_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)
    mcc = matthews_corrcoef(y_train, y_pred)
    auroc = roc_auc_score(y_train, y_pred)
    auprc = average_precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUC': [auroc],
        'AUPRC': [auprc],
    }, index=[col_name])
    return y_pred, metrics

def y_prediction_cv(model, x_train, y_train, col_name):
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)
    y_pred = pd.DataFrame(y_pred, columns=[col_name]).set_index(y_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)
    mcc = matthews_corrcoef(y_train, y_pred)
    auroc = roc_auc_score(y_train, y_pred)
    auprc = average_precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUC': [auroc],
        'AUPRC': [auprc],
    }, index=[col_name])
    return y_pred, metrics


def plot_confusion_matrix_with_acc(y_true, y_pred, class_names, title="Confusion Matrix", filename="confusion_matrix.png"):
    """
    Plots a confusion matrix with per-class accuracy (%) in each box.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = np.round(cm / cm_sum.astype(float) * 100, 1)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(2 + n_classes, 2 + n_classes))
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_ylabel('True label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_title(title, fontsize=12, fontstyle='italic', fontweight='bold')

    # Loop over data dimensions and create text annotations.
    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            perc = cm_perc[i, j]
            ax.text(j, i, f'{value}\n{perc:.1f}%', ha="center", va="center",
                    color="white" if perc > thresh else "black", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs("graph_metrics", exist_ok=True)
    plt.savefig(os.path.join("graph_metrics", filename), dpi=500, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_cv(y_true, y_pred, class_names, title="Confusion Matrix (CV)", filename="confusion_matrix_cv.png"):
    """
    Plots a confusion matrix with per-class accuracy (%) in each box for CV predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = np.round(cm / cm_sum.astype(float) * 100, 1)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(2 + n_classes, 2 + n_classes))
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_ylabel('True label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_title(title, fontsize=12, fontstyle='italic', fontweight='bold')

    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            perc = cm_perc[i, j]
            ax.text(j, i, f'{value}\n{perc:.1f}%', ha="center", va="center",
                    color="white" if perc > thresh else "black", fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs("graph_metrics", exist_ok=True)
    plt.savefig(os.path.join("graph_metrics", filename), dpi=500, bbox_inches='tight')
    plt.close()


def y_random(x_train, x_test, y_train, y_test, metric_train, metric_test, name):
    ACC_test = []
    ACC_train = []
    for i in range(1, 101):
        y_train_shuffled = y_train.sample(frac=1, replace=False, random_state=i)
        model = RandomForestClassifier(max_depth=3, max_features=10, random_state=None).fit(x_train, y_train_shuffled)
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_train = accuracy_score(y_train_shuffled, y_pred_train)
        ACC_test.append(acc_test)
        ACC_train.append(acc_train)
    size = [50]
    sizes = [20]

    # save the metrics
    metrics_df = pd.DataFrame({
        'Accuracy_Train': ACC_train,
        'Accuracy_Test': ACC_test
    })
    metrics_df.to_csv(os.path.join("graph_yrandom", f"Y-randomization-{name}-classification.csv"), index=False)

    # Use the correct index for your metrics DataFrames
    x = [metric_train.loc[name, 'Accuracy']]
    y = [metric_test.loc[name, 'Accuracy']]
    plt.figure(figsize=(4, 3))
    plt.axvline(0.5, c='black', ls=':')
    plt.axhline(0.5, c='black', ls=':')
    plt.scatter(x, y, s=size, c=['red'], marker='x', label='Our model')
    plt.scatter(ACC_train, ACC_test, c='blue', edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    plt.xlabel('Accuracy$_{5-CV}$', fontsize=14, fontstyle='italic', weight='bold')
    plt.ylabel('Accuracy$_{Test}$', fontsize=14, fontstyle='italic', weight='bold')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_yrandom", f"Y-randomization-{name}-classification.png"), bbox_inches='tight', dpi=500)
    plt.close()

def compute_and_plot_shap(model, x_train, x_test, output_prefix, top_n=20, save_dir="graph_shap"):

    os.makedirs(save_dir, exist_ok=True)

    # Create SHAP explainer
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_test, check_additivity=False)

    # Handle classification: shap_values.values is (samples, features, classes)
    if shap_values.values.ndim == 3:
        shap_array = shap_values.values[:, :, 1]  # Use class 1 (Active)
    else:
        shap_array = shap_values.values  # For regression or single output

    # Create DataFrame
    shap_df = pd.DataFrame(shap_array, columns=x_test.columns, index=x_test.index)

    # Save SHAP values
    shap_csv_path = os.path.join(save_dir, f"{output_prefix}_shap_values.csv")
    shap_df.to_csv(shap_csv_path)
    print(f"SHAP values saved to: {shap_csv_path}")

    # Compute mean absolute SHAP
    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)

    # Plot bar chart
    plt.figure(figsize=(5, max(4, 0.3 * top_n)))
    mean_abs_shap.iloc[:top_n].plot(kind='barh', color='skyblue', edgecolor='black')
    plt.xlabel("Mean |SHAP value|", fontsize=12, fontstyle='italic', weight='bold')
    plt.title(f"Top {top_n} SHAP Feature Importances", fontsize=12, fontstyle='italic', weight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{output_prefix}_shap_barplot.png")
    plt.savefig(plot_path, dpi=500)
    plt.close()
    print(f"SHAP bar plot saved to: {plot_path}")


def main():
    all_results = []
    for x in ['AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP', 'Desc', 'ECFP']:
        print("#"*100) 
        x_train = pd.read_csv(os.path.join("trpa1", "train", x+".csv"), index_col=0)
        y_train = pd.read_csv(os.path.join("trpa1", "train", "y_train.csv"), index_col=0)
        x_test  = pd.read_csv(os.path.join("trpa1", "test",  x+".csv"), index_col=0)
        y_test  = pd.read_csv(os.path.join("trpa1", "test",  "y_test.csv"), index_col=0)
        print("Train and test data loaded")
        # Remove constant features
        x_train = remove_constant_string_des(x_train)
        x_train = remove_highly_correlated_features(x_train, threshold=0.7)
        # Fit x_test columns to match x_train
        x_test = x_test[x_train.columns]
        print("Match columns: ",x, "for",  x_train.shape, x_test.shape) 
        # Map labels to numeric values
        label_map = {'Inactive': 0, 'Active': 1}
        y_train = y_train.replace(label_map)
        y_test = y_test.replace(label_map)
        #
        y_train = y_train.infer_objects(copy=False)
        y_test  = y_test.infer_objects(copy=False)
        
        print("y_train count active and inactive:", y_train.value_counts())
        print("y_test count active and inactive:", y_test.value_counts())
        #
        # Squeeze the data to 1D for y_train and y_test to avoid issues
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        
        # Check if the index of x_train and y_train match
        if not x_train.index.equals(y_train.index):
            raise ValueError("Index of x_train and y_train do not match.")
        # Check if the index of x_test and y_test match
        if not x_test.index.equals(y_test.index):
            raise ValueError("Index of x_test and y_test do not match.")
        
        # Scale the data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled  = scaler.transform(x_test)
        # Convert to DataFrame
        x_train_scaled = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns)
        x_test_scaled  = pd.DataFrame(x_test_scaled,  index=x_test.index,  columns=x_test.columns)
        
        print("x_train_scaled shape:", x_train_scaled.shape)
        print("y_train shape:", y_train.shape)
        
        # Train the model
        model = RandomForestClassifier(max_depth=3, max_features=10, random_state=None).fit(x_train_scaled, y_train)
        
        compute_and_plot_shap(model, x_train_scaled, x_test_scaled, output_prefix=f"rf_{x}")
        
        # Predict 5-cv and test
        y_pred_cv, metrics_cv = y_prediction_cv(model, x_train_scaled, y_train, x)
        y_pred_test, metrics_test = y_prediction(model, x_test_scaled, y_test, x)

        # Plot confusion matrix
        plot_confusion_matrix_with_acc(y_test, y_pred_test, ["Inactive", "Active"], title="Confusion Matrix - Test Set", filename=f"confusion_matrix_{x}_test.png")
        plot_confusion_matrix_cv(y_train, y_pred_cv, ["Inactive", "Active"], title="Confusion Matrix - CV", filename=f"confusion_matrix_{x}_cv.png")

        # Y-randomization
        y_random(x_train_scaled, x_test_scaled, y_train, y_test, metrics_cv, metrics_test, x)
        
        # Collect metrics for result_cv.csv
        result_row_cv = {
            "Feature": x,
            "Accuracy": metrics_cv.loc[x, 'Accuracy'],
            "Sensitivity": metrics_cv.loc[x, 'Sensitivity'],
            "Specificity": metrics_cv.loc[x, 'Specificity'],
            "MCC": metrics_cv.loc[x, 'MCC'],
            "AUROC": metrics_cv.loc[x, 'AUC'],
            "AUPRC": metrics_cv.loc[x, 'AUPRC']
        }
        all_results.append(result_row_cv)
        
        # Save results after each fingerprint (append mode)
        results_cv = pd.DataFrame([result_row_cv])
        results_file = "results_cv.csv"

        if os.path.exists(results_file):
            existing = pd.read_csv(results_file)
            results_cv = pd.concat([existing, results_cv], ignore_index=True)

        results_cv.to_csv(results_file, index=False)
        print(f"✅ Results CV appended for {x}")
    
        # Collect metrics for result_test.csv
        result_row = {
            "Feature": x,
            "Accuracy": metrics_test.loc[x, 'Accuracy'],
            "Sensitivity": metrics_test.loc[x, 'Sensitivity'],
            "Specificity": metrics_test.loc[x, 'Specificity'],
            "MCC": metrics_test.loc[x, 'MCC'],
            "AUROC": metrics_test.loc[x, 'AUC'],
            "AUPRC": metrics_test.loc[x, 'AUPRC']
        }
        all_results.append(result_row)

        # Save results after each fingerprint (append mode)
        results_test = pd.DataFrame([result_row])
        results_file = "results_test.csv"

        if os.path.exists(results_file):
            existing = pd.read_csv(results_file)
            results_test = pd.concat([existing, results_test], ignore_index=True)

        results_test.to_csv(results_file, index=False)
        print(f"✅ Results appended for {x}")

        # Save the Random Forest model and scaler
        model_save_path = os.path.join("models", f"{x.lower()}_model.joblib")
        os.makedirs("models", exist_ok=True)
        joblib.dump((model, scaler), model_save_path)
        print(f"\n✅ Model and scaler saved as '{model_save_path}'")

if __name__ == "__main__":
    main()

