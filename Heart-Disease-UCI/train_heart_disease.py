"""
=============================================================================
MediScan Africa — Scénario C : Prédiction des Maladies Cardiovasculaires
Dataset : Heart Disease UCI (Cleveland, 303 patients)
Algorithmes : Logistic Regression vs Random Forest
Explainability : SHAP
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import joblib
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heart_disease_uci.csv')

print("=" * 70)
print("  MediScan Africa — Prédiction des Maladies Cardiovasculaires")
print("  Heart Disease UCI Dataset (Cleveland)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 1. Chargement et nettoyage des données...")

df_raw = pd.read_csv(DATA_PATH)

print(f"   📁 Dataset brut : {df_raw.shape[0]} entrées de {df_raw['dataset'].nunique()} sources")
print(f"      Sources : {df_raw['dataset'].value_counts().to_dict()}")

# ─── Filtrer Cleveland uniquement (303 patients, cahier de charge) ────────────
df = df_raw[df_raw['dataset'] == 'Cleveland'].copy()
print(f"\n   ✅ Filtrage Cleveland : {df.shape[0]} patients retenus")

# Supprimer les colonnes non nécessaires
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# ─── Conversion de la variable cible en binaire ──────────────────────────────
# num : 0 = pas de maladie, 1-4 = maladie cardiaque → binaire 0/1
df['target'] = (df['num'] >= 1).astype(int)
df = df.drop(columns=['num'])

print(f"   📋 Distribution des classes :")
print(f"      - Pas de maladie (0) : {(df['target'] == 0).sum()} ({(df['target'] == 0).mean()*100:.1f}%)")
print(f"      - Maladie cardio (1) : {(df['target'] == 1).sum()} ({(df['target'] == 1).mean()*100:.1f}%)")

# ─── Encodage des variables catégorielles ─────────────────────────────────────

print("\n🔧 Encodage des variables catégorielles...")

# sex : Male/Female → 1/0
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# cp : chest pain type
cp_mapping = {
    'typical angina': 0,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 3
}
df['cp'] = df['cp'].map(cp_mapping)

# restecg : resting ECG
restecg_mapping = {
    'normal': 0,
    'st-t abnormality': 1,
    'lv hypertrophy': 2
}
df['restecg'] = df['restecg'].map(restecg_mapping)

# exang : exercise angina
df['exang'] = df['exang'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

# fbs : fasting blood sugar
df['fbs'] = df['fbs'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

# slope
slope_mapping = {
    'upsloping': 0,
    'flat': 1,
    'downsloping': 2
}
df['slope'] = df['slope'].map(slope_mapping)

# thal
thal_mapping = {
    'normal': 0,
    'fixed defect': 1,
    'reversable defect': 2
}
df['thal'] = df['thal'].map(thal_mapping)

# ─── Gestion des valeurs manquantes ──────────────────────────────────────────

print("\n⚠️  Gestion des valeurs manquantes...")

# Convertir toutes les colonnes en numérique
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

missing_before = df.isnull().sum()
missing_cols = missing_before[missing_before > 0]
if len(missing_cols) > 0:
    print(f"   Colonnes avec valeurs manquantes :")
    for col, count in missing_cols.items():
        print(f"      - {col} : {count} manquantes ({count/len(df)*100:.1f}%)")
    
    # Imputation par la médiane pour les variables numériques
    for col in missing_cols.index:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"      → {col} imputé par médiane = {median_val}")
else:
    print(f"   ✅ Aucune valeur manquante")

print(f"\n   ✅ Dataset final : {df.shape[0]} patients, {df.shape[1] - 1} features")

# ─── Afficher les features ───────────────────────────────────────────────────

feature_names = [c for c in df.columns if c != 'target']
feature_descriptions = {
    'age': 'Âge du patient',
    'sex': 'Sexe (1=Homme, 0=Femme)',
    'cp': 'Type de douleur thoracique (0-3)',
    'trestbps': 'Tension artérielle au repos (mm Hg)',
    'chol': 'Cholestérol sérique (mg/dl)',
    'fbs': 'Glycémie à jeun > 120 mg/dl (1=oui)',
    'restecg': 'ECG au repos (0-2)',
    'thalch': 'Fréquence cardiaque maximale',
    'exang': 'Angine induite par effort (1=oui)',
    'oldpeak': 'Dépression ST induite par effort',
    'slope': 'Pente du segment ST (0-2)',
    'ca': 'Nb de vaisseaux colorés par fluoroscopie',
    'thal': 'Thalassémie (0=normal, 1=fixe, 2=réversible)'
}

print(f"\n   Features ({len(feature_names)}) :")
for i, feat in enumerate(feature_names, 1):
    desc = feature_descriptions.get(feat, '')
    print(f"      {i:2d}. {feat:<12s} — {desc}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING ET NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🔧 2. Normalisation des features...")

X = df.drop('target', axis=1)
y = df['target']

# Division train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Split train/test : {X_train.shape[0]} / {X_test.shape[0]}")

# Normalisation StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarder le scaler
scaler_path = os.path.join(MODELS_DIR, 'heart_disease_scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"   💾 Scaler sauvegardé : {scaler_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT DE 2 ALGORITHMES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🤖 3. Entraînement des modèles...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── 3.1 Logistic Regression ─────────────────────────────────────────────────

print("\n   ── Logistic Regression ──")
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_

print(f"   ✅ Meilleurs paramètres LR : {lr_grid.best_params_}")
print(f"   📈 Meilleur AUC-ROC (CV)  : {lr_grid.best_score_:.4f}")

# ─── 3.2 Random Forest ───────────────────────────────────────────────────────

print("\n   ── Random Forest ──")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_

print(f"   ✅ Meilleurs paramètres RF : {rf_grid.best_params_}")
print(f"   📈 Meilleur AUC-ROC (CV)  : {rf_grid.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION ET COMPARAISON DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 4. Évaluation et comparaison des modèles...")


def compute_medical_metrics(y_true, y_pred, y_proba, model_name):
    """
    Calcule et affiche les métriques médicales essentielles.
    
    En contexte cardiovasculaire :
    - Positif (1) = Maladie cardiovasculaire détectée
    - Négatif (0) = Pas de maladie cardiovasculaire
    
    - VP = patients malades correctement détectés
    - VN = patients sains correctement identifiés
    - FP = patients sains faussement classés comme malades
    - FN = patients malades manqués → DANGER VITAL
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\n   ┌─────────────────────────────────────────────────────┐")
    print(f"   │  📋 MÉTRIQUES — {model_name:^35s} │")
    print(f"   ├─────────────────────────────────────────────────────┤")
    print(f"   │  Accuracy      : {accuracy:.4f}                             │")
    print(f"   │  Precision     : {precision:.4f}  (VPP)                     │")
    print(f"   │  Recall        : {recall:.4f}  (Sensibilité)               │")
    print(f"   │  F1-Score      : {f1:.4f}                                  │")
    print(f"   │  AUC-ROC       : {auc:.4f}                                 │")
    print(f"   ├─────────────────────────────────────────────────────┤")
    print(f"   │  🏥 MÉTRIQUES MÉDICALES                             │")
    print(f"   │  Sensibilité   : {sensitivity:.4f}  (détection des malades) │")
    print(f"   │  Spécificité   : {specificity:.4f}  (exclusion des sains)   │")
    print(f"   │  VPP           : {ppv:.4f}  (si + → % vrai malade)         │")
    print(f"   │  VPN           : {npv:.4f}  (si - → % vrai sain)           │")
    print(f"   ├─────────────────────────────────────────────────────┤")
    print(f"   │  🔢 MATRICE DE CONFUSION MÉDICALE                   │")
    print(f"   │  VP (malades détectés)       : {tp:4d}                      │")
    print(f"   │  VN (sains confirmés)        : {tn:4d}                      │")
    print(f"   │  FP (fausses alertes)        : {fp:4d}                      │")
    print(f"   │  FN (malades manqués ⚠️)     : {fn:4d}                     │")
    print(f"   └─────────────────────────────────────────────────────┘")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'auc': auc, 'sensitivity': sensitivity,
        'specificity': specificity, 'ppv': ppv, 'npv': npv,
        'cm': cm, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# Évaluer Logistic Regression
lr_pred = lr_best.predict(X_test_scaled)
lr_proba = lr_best.predict_proba(X_test_scaled)[:, 1]
lr_metrics = compute_medical_metrics(y_test, lr_pred, lr_proba, "Logistic Regression")

# Évaluer Random Forest
rf_pred = rf_best.predict(X_test_scaled)
rf_proba = rf_best.predict_proba(X_test_scaled)[:, 1]
rf_metrics = compute_medical_metrics(y_test, rf_pred, rf_proba, "Random Forest")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🏆 5. Sélection du meilleur modèle...")

# Critère principal : AUC-ROC (bon équilibre pour les maladies cardio)
if lr_metrics['auc'] > rf_metrics['auc']:
    best_model = lr_best
    best_metrics = lr_metrics
    best_name = "Logistic Regression"
    best_pred = lr_pred
    best_proba = lr_proba
elif rf_metrics['auc'] > lr_metrics['auc']:
    best_model = rf_best
    best_metrics = rf_metrics
    best_name = "Random Forest"
    best_pred = rf_pred
    best_proba = rf_proba
else:
    # Égalité AUC → choisir RF pour SHAP explainability
    best_model = rf_best
    best_metrics = rf_metrics
    best_name = "Random Forest"
    best_pred = rf_pred
    best_proba = rf_proba

print(f"\n   🥇 Meilleur modèle : {best_name}")
print(f"      AUC-ROC   : {best_metrics['auc']:.4f}")
print(f"      Accuracy  : {best_metrics['accuracy']:.4f}")
print(f"      F1-Score  : {best_metrics['f1']:.4f}")

# Comparaison tabulaire
print(f"\n   ┌─────────────────┬──────────────────┬────────────────┐")
print(f"   │ Métrique        │ Logistic Regr.   │ Random Forest  │")
print(f"   ├─────────────────┼──────────────────┼────────────────┤")
print(f"   │ Accuracy        │     {lr_metrics['accuracy']:.4f}         │    {rf_metrics['accuracy']:.4f}        │")
print(f"   │ Precision (VPP) │     {lr_metrics['precision']:.4f}         │    {rf_metrics['precision']:.4f}        │")
print(f"   │ Recall (Sens.)  │     {lr_metrics['recall']:.4f}         │    {rf_metrics['recall']:.4f}        │")
print(f"   │ F1-Score        │     {lr_metrics['f1']:.4f}         │    {rf_metrics['f1']:.4f}        │")
print(f"   │ AUC-ROC         │     {lr_metrics['auc']:.4f}         │    {rf_metrics['auc']:.4f}        │")
print(f"   │ Spécificité     │     {lr_metrics['specificity']:.4f}         │    {rf_metrics['specificity']:.4f}        │")
print(f"   │ VPN             │     {lr_metrics['npv']:.4f}         │    {rf_metrics['npv']:.4f}        │")
print(f"   └─────────────────┴──────────────────┴────────────────┘")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📉 6. Génération des visualisations...")

# ─── 6.1 Matrices de confusion côte à côte ───────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Matrices de Confusion — Maladies Cardiovasculaires (MediScan Africa)',
             fontsize=14, fontweight='bold')

labels = ['Sain', 'Malade']
for ax, metrics, name in zip(axes, [lr_metrics, rf_metrics], ['Logistic Regression', 'Random Forest']):
    cm = metrics['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 14})
    ax.set_xlabel('Prédiction', fontsize=11)
    ax.set_ylabel('Réalité', fontsize=11)
    ax.set_title(f'{name}\nAUC={metrics["auc"]:.3f} | Sens={metrics["sensitivity"]:.3f}', fontsize=11)

plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, 'heart_disease_confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   💾 Matrice de confusion : {cm_path}")

# ─── 6.2 Courbe ROC-AUC ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

for proba, metrics, color, name in [
    (lr_proba, lr_metrics, '#F44336', 'Logistic Regression'),
    (rf_proba, rf_metrics, '#FF9800', 'Random Forest')
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name} (AUC = {metrics["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aléatoire (AUC = 0.500)')
ax.set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
ax.set_title('Courbe ROC — Maladies Cardiovasculaires\n(MediScan Africa)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

roc_path = os.path.join(REPORTS_DIR, 'heart_disease_roc_curve.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   💾 Courbe ROC : {roc_path}")

# ─── 6.3 SHAP Explainability (cahier de charge) ──────────────────────────────

print("\n🔍 7. Analyse SHAP (explainability — cahier de charge)...")

try:
    import shap

    # TreeExplainer pour Random Forest
    explainer = shap.TreeExplainer(rf_best)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 7))
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    shap.summary_plot(shap_vals, X_test_df, show=False, max_display=13)
    plt.title('SHAP — Impact des features sur la prédiction\nMaladies Cardiovasculaires',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    shap_path = os.path.join(REPORTS_DIR, 'heart_disease_shap_summary.png')
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 SHAP summary : {shap_path}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_test_df, plot_type="bar", show=False, max_display=13)
    plt.title('SHAP — Importance moyenne des features\nMaladies Cardiovasculaires',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    shap_bar_path = os.path.join(REPORTS_DIR, 'heart_disease_shap_bar.png')
    plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 SHAP bar plot : {shap_bar_path}")

    # Waterfall plot pour le premier patient du test set (explication individuelle)
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test_df.iloc[0],
            feature_names=feature_names
        ), show=False)
        plt.title('SHAP Waterfall — Explication individuelle (Patient #1)',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        waterfall_path = os.path.join(REPORTS_DIR, 'heart_disease_shap_waterfall.png')
        plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 SHAP waterfall : {waterfall_path}")
    except Exception as e:
        print(f"   ⚠️  Waterfall plot indisponible : {e}")

except ImportError:
    print("   ⚠️  Module 'shap' non installé. Installer avec : pip install shap")
    print("   📌 Génération du feature importance Random Forest à la place...")

    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1], color='#F44336')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance — Random Forest\nMaladies Cardiovasculaires',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(REPORTS_DIR, 'heart_disease_feature_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Feature importance : {fi_path}")

except Exception as e:
    print(f"   ⚠️  Erreur SHAP : {e}")

# ─── 6.4 Coefficients Logistic Regression (interprétabilité) ─────────────────

print("\n📊 Coefficients de la Logistic Regression :")
lr_coefs = pd.Series(lr_best.coef_[0], index=feature_names).sort_values(ascending=False)
print(f"   Facteurs de risque (coefficients positifs = augmente le risque) :")
for feat, coef in lr_coefs.head(5).items():
    print(f"      + {feat:<12s} : {coef:+.4f}")
print(f"   Facteurs protecteurs (coefficients négatifs = diminue le risque) :")
for feat, coef in lr_coefs.tail(5).items():
    print(f"      - {feat:<12s} : {coef:+.4f}")

# Visualisation des coefficients
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#F44336' if c > 0 else '#4CAF50' for c in lr_coefs.values]
ax.barh(range(len(lr_coefs)), lr_coefs.values, color=colors)
ax.set_yticks(range(len(lr_coefs)))
ax.set_yticklabels(lr_coefs.index)
ax.set_xlabel('Coefficient (Logistic Regression)', fontsize=11)
ax.set_title('Facteurs de risque vs protecteurs\nMaladies Cardiovasculaires (Logistic Regression)',
             fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
coef_path = os.path.join(REPORTS_DIR, 'heart_disease_lr_coefficients.png')
plt.savefig(coef_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   💾 Coefficients LR : {coef_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXPORT DU MODÈLE POUR DÉPLOIEMENT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n💾 8. Export du modèle pour déploiement...")

# Sauvegarder le meilleur modèle
best_model_path = os.path.join(MODELS_DIR, 'heart_disease_best_model.joblib')
joblib.dump(best_model, best_model_path)
print(f"   ✅ Meilleur modèle ({best_name}) : {best_model_path}")

# Sauvegarder les deux modèles
lr_model_path = os.path.join(MODELS_DIR, 'heart_disease_lr.joblib')
rf_model_path = os.path.join(MODELS_DIR, 'heart_disease_rf.joblib')
joblib.dump(lr_best, lr_model_path)
joblib.dump(rf_best, rf_model_path)
print(f"   💾 LR sauvegardé : {lr_model_path}")
print(f"   💾 RF sauvegardé : {rf_model_path}")

# Métadonnées
model_metadata = {
    'best_model': best_name,
    'features': feature_names,
    'feature_descriptions': feature_descriptions,
    'n_features': len(feature_names),
    'n_samples_train': X_train.shape[0],
    'n_samples_test': X_test.shape[0],
    'dataset_source': 'Heart Disease UCI — Cleveland (303 patients)',
    'classes': {'0': 'Sain (pas de maladie)', '1': 'Maladie cardiovasculaire'},
    'metrics': {
        'logistic_regression': {k: v for k, v in lr_metrics.items() if k not in ['cm', 'model_name']},
        'random_forest': {k: v for k, v in rf_metrics.items() if k not in ['cm', 'model_name']},
    },
    'best_params': {
        'logistic_regression': lr_grid.best_params_,
        'random_forest': rf_grid.best_params_,
    }
}

metadata_path = os.path.join(MODELS_DIR, 'heart_disease_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=2, ensure_ascii=False, default=str)
print(f"   💾 Metadata : {metadata_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. DÉMONSTRATION : PRÉDICTION SUR 3 PROFILS FICTIFS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🏥 9. Démonstration — 3 profils patients fictifs...")

# Créer 3 profils réalistes
profils_fictifs = [
    {
        'nom': 'Patient A — Homme 55 ans, douleur thoracique asymptomatique',
        'data': {'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 160, 'chol': 289,
                 'fbs': 0, 'restecg': 2, 'thalch': 145, 'exang': 1,
                 'oldpeak': 2.0, 'slope': 1, 'ca': 1, 'thal': 2}
    },
    {
        'nom': 'Patient B — Femme 45 ans, douleur atypique',
        'data': {'age': 45, 'sex': 0, 'cp': 1, 'trestbps': 120, 'chol': 220,
                 'fbs': 0, 'restecg': 0, 'thalch': 175, 'exang': 0,
                 'oldpeak': 0.0, 'slope': 0, 'ca': 0, 'thal': 0}
    },
    {
        'nom': 'Patient C — Homme 65 ans, angine typique',
        'data': {'age': 65, 'sex': 1, 'cp': 0, 'trestbps': 140, 'chol': 254,
                 'fbs': 1, 'restecg': 2, 'thalch': 127, 'exang': 0,
                 'oldpeak': 1.4, 'slope': 1, 'ca': 1, 'thal': 2}
    }
]

for i, profil in enumerate(profils_fictifs, 1):
    sample = pd.DataFrame([profil['data']])[feature_names]
    sample_scaled = scaler.transform(sample)
    
    pred = best_model.predict(sample_scaled)[0]
    proba = best_model.predict_proba(sample_scaled)[0]
    
    risque = "RISQUE ÉLEVÉ ⚠️" if pred == 1 else "FAIBLE RISQUE ✅"
    
    print(f"\n   {profil['nom']} :")
    print(f"      Prédiction      : {risque}")
    print(f"      P(sain)         : {proba[0]*100:.1f}%")
    print(f"      P(maladie)      : {proba[1]*100:.1f}%")
    
    # Recommandation médicale fictive
    if proba[1] > 0.7:
        print(f"      📋 Recommandation : Consultation cardiologique URGENTE")
        print(f"         Examens : ECG, troponine, coronarographie")
    elif proba[1] > 0.4:
        print(f"      📋 Recommandation : Consultation cardiologique programmée")
        print(f"         Examens : ECG d'effort, échocardiographie")
    else:
        print(f"      📋 Recommandation : Surveillance standard")
        print(f"         Examens : Bilan lipidique annuel, tension artérielle")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RAPPORT CLASSIFICATION COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n📄 10. Rapport de classification complet ({best_name}) :")
print(classification_report(y_test, best_pred,
                            target_names=['Sain', 'Maladie Cardio']))

# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"  ✅ ENTRAÎNEMENT TERMINÉ — Modèle sélectionné : {best_name}")
print(f"  📁 Fichiers sauvegardés dans : {OUTPUT_DIR}")
print("=" * 70)
