#!/usr/bin/env python3


import argparse
import hail as hl
import hailtop.batch as hb

TMP_BUCKET = 'gs://aou_tmp/v8'

def extract_vsm_weights(mt, weight_names, filter_missing: bool = True):
    """
    Extract one or more VSM weights and create columns for each.
    
    Parameters:
    -----------
    mt : hail.MatrixTable
        The matrix table with exploded vsm_weights
    weight_names : str or list
        Single weight name or list of weight names to extract
    filter_missing : bool
        Whether to filter out rows where any weight is missing
    
    Returns:
    --------
    hail.MatrixTable
        Matrix table with weight columns
    """
    if isinstance(weight_names, str):
        weight_names = [weight_names]
    
    weight_annotations = {}
    for weight_name in weight_names:
        if len(weight_names) == 1:
            weight_annotations["vsm_weight_value"] = mt.vsm_weights[weight_name]
        else:
            weight_annotations[f"vsm_weight_{weight_name}"] = mt.vsm_weights[weight_name]
    
    mt = mt.annotate_rows(**weight_annotations)
    
    if filter_missing:
        if len(weight_names) == 1:
            mt = mt.filter_rows(hl.is_defined(mt.vsm_weight_value))
        else:
            filter_conditions = [hl.is_defined(mt[f"vsm_weight_{weight_name}"]) for weight_name in weight_names]
            combined_filter = hl.fold(lambda acc, cond: acc & cond, True, filter_conditions)
            mt = mt.filter_rows(combined_filter)
    
    return mt

def create_phenotype_data(mt, phenotype_col: str = None):
    """
    Create binary phenotype from case/control counts or use existing phenotype.
    
    Parameters:
    -----------
    mt : hail.MatrixTable
        The matrix table
    phenotype_col : str, optional
        Column name for existing phenotype data
    
    Returns:
    --------
    hail.MatrixTable
        Matrix table with phenotype annotation
    """
    if phenotype_col:
        return mt
    else:
        mt = mt.filter_rows(
            hl.agg.any((mt['AF.Cases'] == 0) | (mt['AF.Controls'] == 0))
        )
        mt = mt.annotate_entries(
            is_case_variant = hl.if_else(
                (mt['AF.Cases'] == 0) & (mt['AF.Controls'] > 0), False,  # Only in controls → not case variant
                hl.if_else(
                    (mt['AF.Controls'] == 0) & (mt['AF.Cases'] > 0), True,  # Only in cases → case variant
                    hl.missing(hl.tbool)  # Present in both or neither → undefined
                )
            )
        )
        return mt

def prepare_regression_features(mt, weight_names, transformations: list = None, pheno: str = None):
    """
    Prepare feature matrix with various transformations of VSM weights.
    Works with single or multiple VSM weights.
    
    Parameters:
    -----------
    mt : hail.MatrixTable
        The matrix table with exploded vsm_weights
    weight_names : str or list
        Single weight name or list of weight names
    transformations : list, optional
        List of transformations to apply to each weight
    pheno : str, optional
        Existing phenotype column name
        
    Returns:
    --------
    tuple
        (hail.Table, list) - The prepared feature table and list of feature column names
    """
    if transformations is None:
        transformations = ['linear', 'exp', 'square']
    
    if isinstance(weight_names, str):
        weight_names = [weight_names]
        single_mode = True
    else:
        single_mode = False
    
    mt = extract_vsm_weights(mt, weight_names)
    mt = create_phenotype_data(mt, pheno)

    row_feature_exprs = {}

    if single_mode:
        weight_name = weight_names[0]
        weight_col = mt.vsm_weight_value
        
        for tf in transformations:
            row_val = apply_transformation(weight_col, tf)
            row_feature_exprs[f"{weight_name}_{tf}"] = row_val
    else:
        for weight_name in weight_names:
            weight_col = mt[f"vsm_weight_{weight_name}"]
            
            for tf in transformations:
                row_val = apply_transformation(weight_col, tf)
                row_feature_exprs[f"{weight_name}_{tf}"] = row_val

    mt = mt.annotate_rows(**row_feature_exprs)

    entry_feature_exprs = {
        f"{name}_entry": mt[name] for name in row_feature_exprs
    }

    mt = mt.annotate_entries(**entry_feature_exprs)
    ht = mt.entries()
    ht = ht.select(*list(entry_feature_exprs.keys()), 'is_case_variant')
    
    return ht, list(entry_feature_exprs.keys())

def apply_transformation(weight_col, transformation):
    """
    Apply a single transformation to a weight column.
    
    Parameters:
    -----------
    weight_col : hail expression
        The weight column to transform
    transformation : str
        The transformation to apply
        
    Returns:
    --------
    hail expression
        The transformed column
    """
    if transformation == 'linear':
        return weight_col
    elif transformation == 'exp':
        return hl.exp(weight_col)
    elif transformation == 'square':
        return weight_col ** 2
    elif transformation == 'cube':
        return weight_col ** 3
    elif transformation == 'log':
        return hl.log(hl.abs(weight_col) + 1e-8)
    elif transformation == 'abs':
        return hl.abs(weight_col)
    elif transformation == 'sqrt':
        return hl.sqrt(hl.abs(weight_col))
    else:
        raise ValueError(f"Unknown transformation: {transformation}")


def run_all_logistic_models_cv(
    ht,
    feature_cols,
    phenotype_col='is_case_variant',
    min_features=1,
    max_features=None,
    max_models=None,
    cv_folds=5,
    random_state=42,
    single_model_only=False
):
    """
    Fit logistic regression models for all combinations of feature columns,
    use cross-validation, and return top models ranked by AUC.
    
    Parameters:
    -----------
    single_model_only : bool
        If True, only fit one model using ALL features (no combinations)
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from itertools import combinations
    from sklearn.impute import SimpleImputer
    import pandas as pd
    
    df = ht.to_pandas()
    df = df[df[phenotype_col].notnull()]
    y = df[phenotype_col].astype(int)

    all_results = []

    if single_model_only:
        combo = tuple(feature_cols)
        combo_name = '+'.join(combo)
        X = df[list(combo)]
        missing_pct = X.isnull().mean().mean() * 100
        X = X.dropna()
        y = y.loc[X.index]
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        aucs = []
        accs = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accs.append(accuracy_score(y_test, y_pred))
            aucs.append(roc_auc_score(y_test, y_prob))

        avg_acc = np.mean(accs)
        avg_auc = np.mean(aucs)

        final_model = LogisticRegression(random_state=random_state, max_iter=1000)
        final_model.fit(X, y)

        all_results.append({
            'features': combo,
            'feature_key': combo_name,
            'accuracy': avg_acc,
            'auc': avg_auc,
            'coefficients': dict(zip(combo, final_model.coef_[0])),
            'feature_importance': dict(zip(combo, np.abs(final_model.coef_[0]))),
            'missing_pct': missing_pct
        })
        
    else:
        # Single mode: Run combinations (original exploratory behavior)
        if max_features is None:
            max_features = len(feature_cols)

        for k in range(min_features, max_features + 1):
            for combo in combinations(feature_cols, k):
                combo_name = '+'.join(combo)
                X = df[list(combo)]
                missing_pct = X.isnull().mean().mean() * 100 
                X = X.dropna()
                y = y.loc[X.index]
                
                # Cross-validation
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                aucs = []
                accs = []

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model = LogisticRegression(random_state=random_state, max_iter=1000)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    accs.append(accuracy_score(y_test, y_pred))
                    aucs.append(roc_auc_score(y_test, y_prob))

                avg_acc = np.mean(accs)
                avg_auc = np.mean(aucs)

                final_model = LogisticRegression(random_state=random_state, max_iter=1000)
                final_model.fit(X, y)

                all_results.append({
                    'features': combo,
                    'feature_key': combo_name,
                    'accuracy': avg_acc,
                    'auc': avg_auc,
                    'coefficients': dict(zip(combo, final_model.coef_[0])),
                    'feature_importance': dict(zip(combo, np.abs(final_model.coef_[0]))),
                    'missing_pct': missing_pct
                })

        all_results = sorted(all_results, key=lambda r: -r['auc'])

        if max_models is not None:
            all_results = all_results[:max_models]

    return all_results, df


def run_logistic_regression_for_vsm(weight_names, transformations: list, mt_path: str, analysis_mode: str = "single"):
    """
    Unified function for single or multi-VSM logistic regression analysis.
    
    Parameters:
    -----------
    weight_names : str or list
        Single VSM weight name or list of VSM weight names
    transformations : list
        List of transformations to apply
    mt_path : str
        Path to the matrix table
    analysis_mode : str
        'single' for exploratory (combinations), 'multi' for one model with all features
        
    Returns:
    --------
    list
        Results from logistic regression analysis
    """
    import pickle
    
    mt = hl.read_matrix_table(mt_path)
    mt = mt.explode_rows(mt.vsm_weights)

    ht, feature_names = prepare_regression_features(mt, weight_names, transformations=transformations)
    
    single_model_only = (analysis_mode == "multi")
    
    all_res, _ = run_all_logistic_models_cv(ht, feature_names, single_model_only=single_model_only)

    return all_res


def main(args):
    
    backend = hb.ServiceBackend(
        billing_project="all-by-aou", remote_tmpdir=TMP_BUCKET
    )
    b = hb.Batch(name="LogReg VSMs", backend=backend)
    
    mt_path = "gs://aou_amc/tmp/stroke_w_vsms_v2.mt"
    output_bucket = "aou_amc"

    if args.run_logistic_regression:
        if args.analysis_mode == "single":
            for vsm in args.vsms_list:
                j = b.new_python_job(name=f"logreg_{vsm}")
                j.image("hailgenetics/hail:0.2.133-py3.11") 
                j.memory("highmem")
                j.cpu(8)
                j.env('PYSPARK_SUBMIT_ARGS', '--driver-memory 24g --executor-memory 24g pyspark-shell')
                preds = j.call(
                    run_logistic_regression_for_vsm,
                    weight_names=vsm,  # Single string
                    transformations=args.transformations,
                    mt_path=mt_path,
                    output_bucket=output_bucket,
                    analysis_mode="single"
                )
                b.write_output(preds.as_json(), f'gs://{output_bucket}/results/mean_impute/{vsm}_all_res.json')
        
        elif args.analysis_mode == "multi":
            vsm_combo_name = "_".join(args.vsms_list)
            j = b.new_python_job(name=f"logreg_multi_{vsm_combo_name}")
            j.image("hailgenetics/hail:0.2.133-py3.11") 
            j.memory("highmem")
            j.cpu(8)
            j.env('PYSPARK_SUBMIT_ARGS', '--driver-memory 24g --executor-memory 24g pyspark-shell')
            preds = j.call(
                run_logistic_regression_for_vsm,
                weight_names=args.vsms_list,
                transformations=args.transformations,
                mt_path=mt_path,
                analysis_mode="multi"
            )
            b.write_output(preds.as_json(), f'gs://{output_bucket}/results/mean_impute/multi_{vsm_combo_name}_all_res.json')
        
        else:
            raise ValueError(f"Unknown analysis mode: {args.analysis_mode}. Use 'single' or 'multi'.")
        
        b.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--run-logistic-regression",
        help="Run logistic regression analysis",
        action="store_true",
    )
    parser.add_argument(
        "--analysis-mode",
        help="Analysis mode: 'single' for individual VSMs, 'multi' for combined analysis",
        choices=['single', 'multi'],
        default='single'
    )
    parser.add_argument(
        "--vsms-list",
        help="List of VSMs (comma-separated). For 'single' mode: analyzed individually. For 'multi' mode: analyzed together.",
        type=lambda s: s.split(","),
        default=["mpc"]
    )
    parser.add_argument(
        "--transformations",
        help="List of transformations to apply (comma-separated)",
        type=lambda s: s.split(","),
        default=['linear', 'square', 'cube', 'exp']
    )
    
    args = parser.parse_args()
    main(args)
