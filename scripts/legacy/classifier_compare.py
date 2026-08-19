"""SUPERSEDED -- do not use for any reported result.

This script splits with a single `GroupShuffleSplit`. Before `track_id` was
exported to the features CSV, its `detect_group_column()` found no group
column and silently fell back to `train_test_split(stratify=y)` -- a random
split over *segments*. That produced the retracted 0.9051 / 0.8035 figures
(see HANDOFF section 4). The group column now reaches the CSV, so grouping does
engage, but a single shuffle split still holds out ~4 recordings at once
without averaging over folds, and it is not the project's frozen protocol.

Use `run_segment_lr_rf.py` instead. It owns the frozen protocol --
StratifiedGroupKFold, the album group map, segment->track majority vote,
seed 42 -- and `--from-run` re-evaluates a saved run's features.csv.gz under
that protocol without re-extracting. HANDOFF section 6 records the decision not
to extend this file.

Kept only so the retracted figures remain reproducible for the record.
Its outputs live in outputs/legacy/groupshufflesplit/, not in any run's eval/.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Resolved relative to this file (not the caller's cwd), so defaults land in
# <repo root>/outputs/ regardless of where this script is run from.
# Three levels up: scripts/legacy/ -> scripts/ -> repo root.
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any stray index column. Feature columns already carry real names
    (features.csv.gz is written with them), so no renaming table is needed."""
    df = df.copy()
    return df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])


def detect_group_column(df: pd.DataFrame):
    candidates = [
        'track_id', 'recording_id', 'file_id', 'source_file', 'filename',
        'audio_path', 'segment_parent', 'parent_track', 'clip_id'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def split_data(df: pd.DataFrame, feature_cols, label_col='raga_label', test_size=0.25, random_state=42):
    group_col = detect_group_column(df)
    X = df[feature_cols]
    y = df[label_col]

    if group_col is not None:
        groups = df[group_col]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], group_col

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, None


def format_confusion(cm, class_names, normalize=False):
    header = ['true\\pred'] + class_names
    widths = [max(len(h), 12) for h in header]
    for i, cls in enumerate(class_names):
        row_vals = [f'{v:.2f}' if normalize else str(int(v)) for v in cm[i]]
        widths[0] = max(widths[0], len(cls))
        for j, rv in enumerate(row_vals, start=1):
            widths[j] = max(widths[j], len(rv))

    def fmt_row(vals):
        return '  '.join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

    lines = [fmt_row(header)]
    for i, cls in enumerate(class_names):
        row_vals = [f'{v:.2f}' if normalize else str(int(v)) for v in cm[i]]
        lines.append(fmt_row([cls] + row_vals))
    return '\n'.join(lines)


def top_logreg_features(model, feature_names, class_names, top_n=12):
    coefs = model.named_steps['clf'].coef_
    # Binary problems give coef_ shape (1, n_features), not (n_classes, ...),
    # so the second class shares the first row with the sign flipped.
    rows = []
    for class_idx, class_name in enumerate(class_names):
        coef = coefs[0] * (1 if class_idx == 0 or len(coefs) > 1 else -1) \
            if len(coefs) == 1 else coefs[class_idx]
        order = np.argsort(np.abs(coef))[::-1][:top_n]
        rows.append((class_name, [(feature_names[i], float(coef[i])) for i in order]))
    return rows


def top_rf_features(model, feature_names, top_n=20):
    importances = model.named_steps['clf'].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in order]


def write_logreg_report(path, model, X_train, X_test, y_train, y_test, class_names, feature_names, group_col):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_test, y_pred, labels=class_names, normalize='true')
    top_features = top_logreg_features(model, feature_names, class_names)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Logistic Regression Analysis\n')
        f.write('============================\n\n')
        f.write('Setup\n-----\n')
        f.write(f'- Model: Multinomial logistic regression with L2 regularization\n')
        f.write(f'- Train segments: {len(X_train)}\n')
        f.write(f'- Test segments: {len(X_test)}\n')
        f.write(f'- Number of classes: {len(class_names)}\n')
        f.write(f'- Number of features: {len(feature_names)}\n')
        f.write(f'- Group-aware split used: {"yes (" + group_col + ")" if group_col else "no; stratified segment split"}\n\n')

        f.write('Accuracy\n--------\n')
        f.write(f'- Segment-level accuracy: {acc:.4f}\n\n')

        f.write('Classification report\n---------------------\n')
        f.write(classification_report(y_test, y_pred, digits=4)) # type: ignore
        f.write('\n')

        f.write('Confusion matrix\n----------------\n')
        f.write(format_confusion(cm, class_names, normalize=False))
        f.write('\n\n')

        f.write('Confusion proportions\n---------------------\n')
        f.write(format_confusion(cm_norm, class_names, normalize=True))
        f.write('\n\n')

        f.write('Most influential features by class\n---------------------------------\n')
        for class_name, feats in top_features:
            f.write(f'\n{class_name}\n')
            for feat_name, weight in feats:
                direction = 'positive' if weight > 0 else 'negative'
                f.write(f'- {feat_name}: {weight:.6f} ({direction} weight)\n')

        f.write('\nInterpretation notes\n--------------------\n')
        f.write('- Logistic regression is a linear classifier, so it tests whether the current feature set supports reasonably separable class regions with linear decision boundaries.\n')
        f.write('- If this performs similarly to or better than KNN, the features are carrying class information in a globally organized way rather than only through local neighbourhood structure.\n')
        f.write('- If this underperforms KNN, then local clusters matter more than a single global boundary.\n')
        f.write('- Large positive or negative class weights indicate which features the model leans on most strongly for that class, but the sign should be interpreted relative to all other features after standardization.\n')


def write_rf_report(path, model, X_train, X_test, y_train, y_test, class_names, feature_names, group_col):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_test, y_pred, labels=class_names, normalize='true')
    top_features = top_rf_features(model, feature_names)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Random Forest Analysis\n')
        f.write('======================\n\n')
        f.write('Setup\n-----\n')
        f.write(f'- Model: Random forest classifier\n')
        f.write(f'- Train segments: {len(X_train)}\n')
        f.write(f'- Test segments: {len(X_test)}\n')
        f.write(f'- Number of classes: {len(class_names)}\n')
        f.write(f'- Number of features: {len(feature_names)}\n')
        f.write(f'- Group-aware split used: {"yes (" + group_col + ")" if group_col else "no; stratified segment split"}\n\n')

        f.write('Accuracy\n--------\n')
        f.write(f'- Segment-level accuracy: {acc:.4f}\n\n')

        f.write('Classification report\n---------------------\n')
        f.write(classification_report(y_test, y_pred, digits=4)) # type: ignore
        f.write('\n')

        f.write('Confusion matrix\n----------------\n')
        f.write(format_confusion(cm, class_names, normalize=False))
        f.write('\n\n')

        f.write('Confusion proportions\n---------------------\n')
        f.write(format_confusion(cm_norm, class_names, normalize=True))
        f.write('\n\n')

        f.write('Global feature importance\n-------------------------\n')
        for feat_name, importance in top_features:
            f.write(f'- {feat_name}: {importance:.6f}\n')

        f.write('\nInterpretation notes\n--------------------\n')
        f.write('- Random forest can model non-linear interactions and threshold effects, so it is a good next comparison against both KNN and logistic regression.\n')
        f.write('- If this beats logistic regression by a meaningful margin, the feature space likely contains useful non-linear structure.\n')
        f.write('- If this still struggles, then the limitation is more likely in the representation or the intrinsic overlap of classes rather than the classifier family alone.\n')
        f.write('- Feature importance here is global and does not show direction; it only shows how much each feature contributes to reducing impurity across the forest.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True,
                        help="run directory name under outputs/runs/ (or a full path)")
    parser.add_argument('--outdir', default=None,
                        help="defaults to <run>/eval/")
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--rf-estimators', type=int, default=300)
    parser.add_argument('--rf-max-depth', type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = RUNS_DIR / args.run
    features = run_dir / "features.csv.gz"
    if not features.exists():
        raise SystemExit(f"no features.csv.gz in {run_dir}")

    outdir = Path(args.outdir) if args.outdir else run_dir / "eval"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features)
    df = clean_dataframe(df)
    if 'raga_label' not in df.columns:
        raise ValueError('CSV must contain a raga_label column.')

    feature_cols = [c for c in df.columns if c != 'raga_label' and pd.api.types.is_numeric_dtype(df[c])]
    X_train, X_test, y_train, y_test, group_col = split_data(
        df, feature_cols, test_size=args.test_size, random_state=args.random_state
    )

    class_names = sorted(df['raga_label'].unique().tolist())

    logreg = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=5000,
            # multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            random_state=args.random_state
        ))
    ])

    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
            class_weight='balanced_subsample'
        ))
    ])

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Same naming convention as run_segment_lr_rf.py's eval/ output, so a run's
    # eval/ stays readable when several tools write into it.
    stem = f"{len(feature_cols)}feat-full_by-{group_col or 'none'}_groupshufflesplit"
    write_logreg_report(
        outdir / f'{stem}_logreg.txt',
        logreg, X_train, X_test, y_train, y_test, class_names, feature_cols, group_col
    )
    write_rf_report(
        outdir / f'{stem}_rf.txt',
        rf, X_train, X_test, y_train, y_test, class_names, feature_cols, group_col
    )

    summary = {
        'train_segments': len(X_train),
        'test_segments': len(X_test),
        'classes': class_names,
        'features': feature_cols,
        'group_col': group_col,
        'logreg_accuracy': float(accuracy_score(y_test, logreg.predict(X_test))),
        'rf_accuracy': float(accuracy_score(y_test, rf.predict(X_test))),
    }
    with open(outdir / f'{stem}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Done.')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
