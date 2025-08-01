#!/usr/bin/env python3


import hail as hl
import argparse
import hailtop.batch as hb

TMP_BUCKET = 'gs://aou_tmp/v8'

def extract_vsm_weights(mt, weight_name: str, filter_missing: bool = True):
    """
    Explode vsm_weights so each transcript gets its own row,
    and extract the desired weight field as a scalar.
    """    
    # Extract the specific weight from the now-flat struct
    mt = mt.annotate_rows(
        vsm_weight_value=mt.vsm_weights[weight_name]
    )
    
    if filter_missing:
        mt = mt.filter_rows(hl.is_defined(mt.vsm_weight_value))
    
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
        # Use existing phenotype column
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

def prepare_regression_features(mt, weight_name: str, transformations: list = None, pheno: str = None):
    """
    Prepare feature matrix with various transformations of scalar VSM weights,
    and broadcast these row-level features to entry-level so they can be used in
    logistic_regression_rows.
    """
    if transformations is None:
        transformations = ['linear', 'exp', 'square']
    
    # Extract the weights and create the phenotype entry
    mt = extract_vsm_weights(mt, weight_name)
    mt = create_phenotype_data(mt, pheno)

    # Step 1: Add row-level transformed features
    row_feature_exprs = {}

    for tf in transformations:
        if tf == 'linear':
            row_val = mt.vsm_weight_value
        elif tf == 'exp':
            row_val = hl.exp(mt.vsm_weight_value)
        elif tf == 'square':
            row_val = mt.vsm_weight_value ** 2
        elif tf == 'cube':
            row_val = mt.vsm_weight_value** 3
        elif tf == 'log':
            row_val = hl.log(hl.abs(mt.vsm_weight_value) + 1e-8)
        elif tf == 'abs':
            row_val = hl.abs(mt.vsm_weight_value)
        elif tf == 'sqrt':
            row_val = hl.sqrt(hl.abs(mt.vsm_weight_value))
        else:
            raise ValueError(f"Unknown transformation: {tf}")
        
        row_feature_exprs[f"{weight_name}_{tf}"] = row_val

    mt = mt.annotate_rows(**row_feature_exprs)

    # Step 2: Broadcast row fields to entries with different names
    entry_feature_exprs = {
        f"{name}_entry": mt[name] for name in row_feature_exprs
    }

    mt = mt.annotate_entries(**entry_feature_exprs)
    ht = mt.entries()
    ht = ht.select(*list(entry_feature_exprs.keys()), 'is_case_variant')
    
    return ht, list(entry_feature_exprs.keys())


def run_all_logistic_models_cv(
    ht,
    feature_cols,
    phenotype_col='is_case_variant',
    min_features=1,
    max_features=None,
    max_models=None,
    cv_folds=5,
    random_state=42
):
    """
    Fit logistic regression models for all combinations of feature columns,
    use cross-validation, and return top models ranked by AUC.
    """
    # Import sklearn modules inside the function
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from itertools import combinations
    
    df = ht.to_pandas()
    df = df[df[phenotype_col].notnull()]
    y = df[phenotype_col].astype(int)

    if max_features is None:
        max_features = len(feature_cols)

    all_results = []

    for k in range(min_features, max_features + 1):
        for combo in combinations(feature_cols, k):
            combo_name = '+'.join(combo)
            X = df[list(combo)].fillna(0)
            
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
                'model': final_model,
                'accuracy': avg_acc,
                'auc': avg_auc,
                'coefficients': dict(zip(combo, final_model.coef_[0])),
                'feature_importance': dict(zip(combo, np.abs(final_model.coef_[0]))),
            })

    all_results = sorted(all_results, key=lambda r: -r['auc'])

    if max_models is not None:
        all_results = all_results[:max_models]

    return all_results, df


def run_logistic_regression_for_vsm(vsm: str, transformations: list, mt_path: str, output_bucket: str):
    import pickle
    
    mt = hl.read_matrix_table(mt_path)
    mt = mt.explode_rows(mt.vsm_weights)

    ht, feature_names = prepare_regression_features(mt, vsm, transformations=transformations)
    all_res, _ = run_all_logistic_models_cv(ht, feature_names)

    local_file = f"{vsm}_all_res.pkl"
    with open(local_file, "wb") as f:
        pickle.dump(all_res, f)
    
    return local_file


def main(args):
    
    backend = hb.ServiceBackend(
        billing_project="all-by-aou", remote_tmpdir=TMP_BUCKET
    )
    b = hb.Batch(name="LogReg VSMs", backend=backend)
    
    mt_path = "gs://aou_amc/tmp/stroke_w_vsms_v2.mt"
    output_bucket = "aou_amc"

    if args.run_logistic_regression:
        for vsm in args.vsms_list:
            j = b.new_python_job(name=f"logreg_{vsm}")
            j.image("hailgenetics/hail:0.2.133-py3.11") 
            j.memory("highmem")
            j.cpu(8)
            j.env('PYSPARK_SUBMIT_ARGS', '--driver-memory 24g --executor-memory 24g pyspark-shell')
            local_pkl_file = j.call(
                run_logistic_regression_for_vsm,
                vsm=vsm,
                transformations=['linear', 'square', 'cube', 'exp'],
                mt_path=mt_path,
                output_bucket=output_bucket
            )
            b.write_output(local_pkl_file, f'gs://{output_bucket}/results/{vsm}_all_res.pkl')
        
        b.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--run-logistic-regression",
        help="Force run of results loading",
        action="store_true",
    )
    parser.add_argument(
        "--vsms-list",
        help="All the VSMs to test",
        type=lambda s: s.split(","),
        default=["mpc"]
    )
    
    args = parser.parse_args()
    main(args)

