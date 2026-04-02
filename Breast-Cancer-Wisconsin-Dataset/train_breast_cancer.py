"""
=============================================================================
MediScan Africa — Scénario B : Détection du Cancer du Sein
Dataset : Breast Cancer Wisconsin Diagnostic (569 cas, 30 features)
Algorithmes : SVM (RBF) vs Random Forest
Optimisation : Precision (minimiser les faux positifs)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
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
    make_scorer
)
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')

print("=" * 70)
print("  MediScan Africa — Détection du Cancer du Sein")
print("  Breast Cancer Wisconsin Diagnostic Dataset")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 1. Chargement et nettoyage des données...")

df = pd.read_csv(DATA_PATH)

# Supprimer la colonne 'id' (non pertinente) et la colonne vide (dernière)
# La dernière colonne du CSV a un header vide → colonne 'Unnamed: 32'
cols_to_drop = ['id']
unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
cols_to_drop.extend(unnamed_cols)
df = df.drop(columns=cols_to_drop, errors='ignore')

# Encoder la variable cible : M (malin) = 1, B (bénin) = 0
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # B=0, M=1

print(f"   ✅ Dataset chargé : {df.shape[0]} échantillons, {df.shape[1] - 1} features")
print(f"   📋 Distribution des classes :")
print(f"      - Bénin (B=0)  : {(df['diagnosis'] == 0).sum()} ({(df['diagnosis'] == 0).mean()*100:.1f}%)")
print(f"      - Malin (M=1)  : {(df['diagnosis'] == 1).sum()} ({(df['diagnosis'] == 1).mean()*100:.1f}%)")

# Vérifier les valeurs manquantes
missing = df.isnull().sum().sum()
print(f"   ⚠️  Valeurs manquantes : {missing}")
if missing > 0:
    df = df.dropna()
    print(f"   ✅ Après suppression : {df.shape[0]} échantillons")

print(f"\n   Features ({df.shape[1] - 1}) :")
feature_names = [c for c in df.columns if c != 'diagnosis']
for i, feat in enumerate(feature_names, 1):
    print(f"      {i:2d}. {feat}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING ET NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🔧 2. Feature engineering et normalisation...")

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Division train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Split train/test : {X_train.shape[0]} / {X_test.shape[0]}")

# Normalisation StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarder le scaler pour déploiement
scaler_path = os.path.join(MODELS_DIR, 'breast_cancer_scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"   💾 Scaler sauvegardé : {scaler_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT DE 2 ALGORITHMES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🤖 3. Entraînement des modèles...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
precision_scorer = make_scorer(precision_score)

# ─── 3.1 SVM (Support Vector Machine) ────────────────────────────────────────

print("\n   ── SVM (Support Vector Machine) ──")
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv=cv,
    scoring='precision',  # Optimiser la precision
    n_jobs=-1,
    verbose=0
)
svm_grid.fit(X_train_scaled, y_train)
svm_best = svm_grid.best_estimator_

print(f"   ✅ Meilleurs paramètres SVM : {svm_grid.best_params_}")
print(f"   📈 Meilleure precision (CV) : {svm_grid.best_score_:.4f}")

# ─── 3.2 Random Forest ───────────────────────────────────────────────────────

print("\n   ── Random Forest ──")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv,
    scoring='precision',  # Optimiser la precision
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_

print(f"   ✅ Meilleurs paramètres RF  : {rf_grid.best_params_}")
print(f"   📈 Meilleure precision (CV) : {rf_grid.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION ET COMPARAISON DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📊 4. Évaluation et comparaison des modèles...")


def compute_medical_metrics(y_true, y_pred, y_proba, model_name):
    """
    Calcule et affiche les métriques médicales essentielles.
    
    En contexte médical :
    - Positif (1) = Malin (cancer détecté)
    - Négatif (0) = Bénin (pas de cancer)
    
    - VP (Vrais Positifs)  = cancers correctement détectés
    - VN (Vrais Négatifs)  = bénins correctement identifiés
    - FP (Faux Positifs)   = bénins faussement classés comme malins → ANXIÉTÉ PATIENT
    - FN (Faux Négatifs)   = cancers manqués → DANGER VITAL
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métriques médicales
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Rappel / Sensibilité
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Spécificité
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0           # Valeur Prédictive Positive
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0           # Valeur Prédictive Négative
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\n   ┌─────────────────────────────────────────────────────┐")
    print(f"   │  📋 MÉTRIQUES — {model_name:^35s} │")
    print(f"   ├─────────────────────────────────────────────────────┤")
    print(f"   │  Accuracy      : {accuracy:.4f}                             │")
    print(f"   │  Precision     : {precision:.4f}  (VPP — min. faux positifs) │")
    print(f"   │  Recall        : {recall:.4f}  (Sensibilité)               │")
    print(f"   │  F1-Score      : {f1:.4f}                                  │")
    print(f"   │  AUC-ROC       : {auc:.4f}                                 │")
    print(f"   ├─────────────────────────────────────────────────────┤")
    print(f"   │  🏥 MÉTRIQUES MÉDICALES                             │")
    print(f"   │  Sensibilité   : {sensitivity:.4f}  (taux de détection)     │")
    print(f"   │  Spécificité   : {specificity:.4f}  (taux de vrais négatifs)│")
    print(f"   │  VPP           : {ppv:.4f}  (si positif → % vrai cancer)   │")
    print(f"   │  VPN           : {npv:.4f}  (si négatif → % vrai bénin)    │")
    print(f"   ├─────────────────────────────────────────────────────┤")
    print(f"   │  🔢 MATRICE DE CONFUSION MÉDICALE                   │")
    print(f"   │  VP (cancers détectés)       : {tp:4d}                      │")
    print(f"   │  VN (bénins confirmés)       : {tn:4d}                      │")
    print(f"   │  FP (fausses alertes cancer) : {fp:4d}                      │")
    print(f"   │  FN (cancers manqués ⚠️)     : {fn:4d}                     │")
    print(f"   └─────────────────────────────────────────────────────┘")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'auc': auc, 'sensitivity': sensitivity,
        'specificity': specificity, 'ppv': ppv, 'npv': npv,
        'cm': cm, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# Évaluer SVM
svm_pred = svm_best.predict(X_test_scaled)
svm_proba = svm_best.predict_proba(X_test_scaled)[:, 1]
svm_metrics = compute_medical_metrics(y_test, svm_pred, svm_proba, "SVM (RBF)")

# Évaluer Random Forest
rf_pred = rf_best.predict(X_test_scaled)
rf_proba = rf_best.predict_proba(X_test_scaled)[:, 1]
rf_metrics = compute_medical_metrics(y_test, rf_pred, rf_proba, "Random Forest")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🏆 5. Sélection du meilleur modèle...")

# Critère : meilleure precision (objectif du cahier de charge)
# En cas d'égalité, on préfère le modèle avec meilleur AUC
if svm_metrics['precision'] > rf_metrics['precision']:
    best_model = svm_best
    best_metrics = svm_metrics
    best_name = "SVM"
elif rf_metrics['precision'] > svm_metrics['precision']:
    best_model = rf_best
    best_metrics = rf_metrics
    best_name = "Random Forest"
else:
    # Égalité en precision → choisir par AUC
    if svm_metrics['auc'] >= rf_metrics['auc']:
        best_model = svm_best
        best_metrics = svm_metrics
        best_name = "SVM"
    else:
        best_model = rf_best
        best_metrics = rf_metrics
        best_name = "Random Forest"

print(f"\n   🥇 Meilleur modèle : {best_name}")
print(f"      Precision : {best_metrics['precision']:.4f}")
print(f"      AUC-ROC   : {best_metrics['auc']:.4f}")
print(f"      Recall    : {best_metrics['recall']:.4f}")

# Comparaison tabulaire
print(f"\n   ┌─────────────────┬────────────┬────────────────┐")
print(f"   │ Métrique        │    SVM     │ Random Forest  │")
print(f"   ├─────────────────┼────────────┼────────────────┤")
print(f"   │ Accuracy        │  {svm_metrics['accuracy']:.4f}    │    {rf_metrics['accuracy']:.4f}        │")
print(f"   │ Precision (VPP) │  {svm_metrics['precision']:.4f}    │    {rf_metrics['precision']:.4f}        │")
print(f"   │ Recall (Sens.)  │  {svm_metrics['recall']:.4f}    │    {rf_metrics['recall']:.4f}        │")
print(f"   │ F1-Score        │  {svm_metrics['f1']:.4f}    │    {rf_metrics['f1']:.4f}        │")
print(f"   │ AUC-ROC         │  {svm_metrics['auc']:.4f}    │    {rf_metrics['auc']:.4f}        │")
print(f"   │ Spécificité     │  {svm_metrics['specificity']:.4f}    │    {rf_metrics['specificity']:.4f}        │")
print(f"   │ VPN             │  {svm_metrics['npv']:.4f}    │    {rf_metrics['npv']:.4f}        │")
print(f"   └─────────────────┴────────────┴────────────────┘")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📉 6. Génération des visualisations...")

# ─── 6.1 Matrices de confusion côte à côte ───────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Matrices de Confusion — Cancer du Sein (MediScan Africa)', fontsize=14, fontweight='bold')

labels = ['Bénin (B)', 'Malin (M)']
for ax, metrics, name in zip(axes, [svm_metrics, rf_metrics], ['SVM (RBF)', 'Random Forest']):
    cm = metrics['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 14})
    ax.set_xlabel('Prédiction', fontsize=11)
    ax.set_ylabel('Réalité', fontsize=11)
    ax.set_title(f'{name}\nPrecision={metrics["precision"]:.3f} | Recall={metrics["recall"]:.3f}', fontsize=11)

plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, 'breast_cancer_confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   💾 Matrice de confusion : {cm_path}")

# ─── 6.2 Courbe ROC-AUC ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

for proba, metrics, color, name in [
    (svm_proba, svm_metrics, '#2196F3', 'SVM (RBF)'),
    (rf_proba, rf_metrics, '#4CAF50', 'Random Forest')
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name} (AUC = {metrics["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aléatoire (AUC = 0.500)')
ax.set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
ax.set_title('Courbe ROC — Détection Cancer du Sein\n(MediScan Africa)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

roc_path = os.path.join(REPORTS_DIR, 'breast_cancer_roc_curve.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   💾 Courbe ROC : {roc_path}")

# ─── 6.3 SHAP (Interprétabilité) ─────────────────────────────────────────────

print("\n🔍 7. Analyse SHAP (interprétabilité)...")

try:
    import shap

    # Utiliser TreeExplainer pour Random Forest (plus rapide)
    explainer = shap.TreeExplainer(rf_best)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Pour classification binaire, shap_values est une liste [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Classe positive (Malin)
    else:
        shap_vals = shap_values

    shap.summary_plot(shap_vals, X_test_df, show=False, max_display=15)
    plt.title('SHAP — Features les plus importantes\npour la détection du cancer du sein', fontsize=12, fontweight='bold')
    plt.tight_layout()
    shap_path = os.path.join(REPORTS_DIR, 'breast_cancer_shap_summary.png')
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 SHAP summary : {shap_path}")

    # Feature importance bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_df, plot_type="bar", show=False, max_display=15)
    plt.title('SHAP — Importance des features (bar)\npour la détection du cancer du sein', fontsize=12, fontweight='bold')
    plt.tight_layout()
    shap_bar_path = os.path.join(REPORTS_DIR, 'breast_cancer_shap_bar.png')
    plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 SHAP bar plot : {shap_bar_path}")

except ImportError:
    print("   ⚠️  Module 'shap' non installé. Installer avec : pip install shap")
    print("   📌 L'interprétabilité SHAP sera disponible après installation.")
except Exception as e:
    print(f"   ⚠️  Erreur SHAP : {e}")
    print("   📌 Utilisation du feature importance du Random Forest à la place.")

    # Fallback : feature importance du Random Forest
    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1], color='#2196F3')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance — Random Forest\nDétection du cancer du sein', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(REPORTS_DIR, 'breast_cancer_feature_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Feature importance : {fi_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXPORT DU MODÈLE POUR DÉPLOIEMENT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n💾 8. Export du modèle pour déploiement...")

# Sauvegarder le meilleur modèle
best_model_path = os.path.join(MODELS_DIR, 'breast_cancer_best_model.joblib')
joblib.dump(best_model, best_model_path)
print(f"   ✅ Meilleur modèle ({best_name}) : {best_model_path}")

# Sauvegarder aussi les deux modèles pour comparaison
svm_model_path = os.path.join(MODELS_DIR, 'breast_cancer_svm.joblib')
rf_model_path = os.path.join(MODELS_DIR, 'breast_cancer_rf.joblib')
joblib.dump(svm_best, svm_model_path)
joblib.dump(rf_best, rf_model_path)
print(f"   💾 SVM sauvegardé : {svm_model_path}")
print(f"   💾 RF sauvegardé  : {rf_model_path}")

# Sauvegarder les métadonnées du modèle
model_metadata = {
    'best_model': best_name,
    'features': feature_names,
    'n_features': len(feature_names),
    'n_samples_train': X_train.shape[0],
    'n_samples_test': X_test.shape[0],
    'classes': {'0': 'Benin', '1': 'Malin'},
    'metrics': {
        'svm': {k: v for k, v in svm_metrics.items() if k not in ['cm', 'model_name']},
        'rf': {k: v for k, v in rf_metrics.items() if k not in ['cm', 'model_name']},
    },
    'best_params': {
        'svm': svm_grid.best_params_,
        'rf': rf_grid.best_params_,
    }
}

import json
metadata_path = os.path.join(MODELS_DIR, 'breast_cancer_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=2, ensure_ascii=False, default=str)
print(f"   💾 Metadata : {metadata_path}")

# Sauvegarder le label encoder
le_path = os.path.join(MODELS_DIR, 'breast_cancer_label_encoder.joblib')
joblib.dump(le, le_path)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. DÉMONSTRATION : PRÉDICTION SUR 3 PROFILS FICTIFS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🏥 9. Démonstration — 3 profils patients fictifs...")

# Utiliser les 3 premiers éléments du test set comme profils de démonstration
demo_indices = [0, 1, 2]
X_demo = X_test_scaled[demo_indices]
y_demo = y_test.iloc[demo_indices].values

for i, idx in enumerate(demo_indices, 1):
    sample = X_demo[i-1].reshape(1, -1)
    pred = best_model.predict(sample)[0]
    proba = best_model.predict_proba(sample)[0]
    
    real = "MALIN ⚠️" if y_demo[i-1] == 1 else "BÉNIN ✅"
    predicted = "MALIN ⚠️" if pred == 1 else "BÉNIN ✅"
    
    print(f"\n   Patient fictif #{i} :")
    print(f"      Diagnostic réel      : {real}")
    print(f"      Prédiction modèle    : {predicted}")
    print(f"      Probabilité bénin    : {proba[0]*100:.1f}%")
    print(f"      Probabilité malin    : {proba[1]*100:.1f}%")
    print(f"      {'✅ CORRECT' if pred == y_demo[i-1] else '❌ ERREUR'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RAPPORT CLASSIFICATION COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n📄 10. Rapport de classification complet ({best_name}) :")
print(classification_report(y_test, best_model.predict(X_test_scaled),
                            target_names=['Bénin (B)', 'Malin (M)']))

# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"  ✅ ENTRAÎNEMENT TERMINÉ — Modèle sélectionné : {best_name}")
print(f"  📁 Fichiers sauvegardés dans : {OUTPUT_DIR}")
print("=" * 70)
