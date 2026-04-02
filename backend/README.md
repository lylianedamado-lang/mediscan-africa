# MediScan Africa — Backend (FastAPI)

Ce répertoire contient l'API de diagnostic pour le projet P-10.

## 🚀 Installation Locale

1.  **Créer un environnement virtuel** :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

2.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer l'API** :
    ```bash
    python main.py
    ```

## 🔍 Test des Endpoints

Une fois lancé, accédez à la documentation interactive :
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

### Exemple de requête (Cancer) :
```json
POST /diagnostic/cancer
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  ... (les 30 features)
}
```

## 📁 Structure
- `main.py` : Application FastAPI.
- `models/` : Contient les modèles `.joblib` et scalers.
- `requirements.txt` : Dépendances Python.
