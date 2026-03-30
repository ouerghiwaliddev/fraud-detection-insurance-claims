"""
Main script for Insurance Fraud Detection analysis and modeling.
"""

import argparse
import sys
from data_utils import load_data, preprocess_data, scale_features
from model_utils import (split_data, train_knn, train_logistic_regression, train_decision_tree,
                         train_random_forest, train_xgboost, evaluate_model, save_model_artifacts)
from visualization_utils import (setup_plot_style, plot_target_distribution, plot_correlation_matrix,
                                 plot_feature_importance, plot_model_comparison)


def main():
    parser = argparse.ArgumentParser(description='Insurance Fraud Detection Analysis')
    parser.add_argument('--action', choices=['train', 'analyze', 'compare'], default='train',
                       help='Action to perform')
    parser.add_argument('--model', choices=['knn', 'logistic', 'decision_tree', 'random_forest', 'xgboost'],
                       default='xgboost', help='Model to train')
    parser.add_argument('--data_path', default='insurance_claims.csv', help='Path to dataset')

    args = parser.parse_args()

    # Setup plotting
    setup_plot_style()

    # Load and preprocess data
    print("Loading data...")
    df = load_data(args.data_path)
    print(f"Dataset shape: {df.shape}")

    X, y, le_sex, feature_names = preprocess_data(df)
    print(f"Features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Get categorical mappings
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_mappings = {}
    for col in cat_cols:
        ohe_cols = [c for c in feature_names if c.startswith(col + "_")]
        cat_mappings[col] = ohe_cols

    if args.action == 'train':
        print(f"Training {args.model} model...")

        if args.model == 'knn':
            model = train_knn(X_train_scaled, y_train)
        elif args.model == 'logistic':
            model = train_logistic_regression(X_train_scaled, y_train)
        elif args.model == 'decision_tree':
            model = train_decision_tree(X_train_scaled, y_train)
        elif args.model == 'random_forest':
            model = train_random_forest(X_train_scaled, y_train)
        elif args.model == 'xgboost':
            model = train_xgboost(X_train_scaled, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test_scaled, y_test)
        print("Model Metrics:")
        for key, value in metrics.items():
            if key not in ['confusion_matrix', 'classification_report']:
                print(f"  {key}: {value:.4f}")

        # Save artifacts
        save_model_artifacts(model, scaler, le_sex, feature_names, cat_mappings)
        print("Model saved!")

        # Plot feature importance if applicable
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names)

    elif args.action == 'analyze':
        print("Performing data analysis...")

        # Plot target distribution
        plot_target_distribution(y.values)

        # Plot correlation matrix
        plot_correlation_matrix(df)

    elif args.action == 'compare':
        print("Comparing models...")

        models = {
            'KNN': train_knn(X_train_scaled, y_train),
            'Logistic Regression': train_logistic_regression(X_train_scaled, y_train),
            'Decision Tree': train_decision_tree(X_train_scaled, y_train),
            'Random Forest': train_random_forest(X_train_scaled, y_train),
            'XGBoost': train_xgboost(X_train_scaled, y_train)
        }

        model_metrics = {}
        for name, model in models.items():
            metrics = evaluate_model(model, X_test_scaled, y_test)
            model_metrics[name] = metrics
            print(f"{name} F1-Score: {metrics['f1_score']:.4f}")

        # Plot comparison
        plot_model_comparison(model_metrics, 'f1_score')


if __name__ == "__main__":
    main()