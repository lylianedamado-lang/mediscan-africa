import json
import joblib
import boto3
import os
import pandas as pd
import io

# ─── Configuration ───────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get('S3_BUCKET', 'mediscan-africa-storage')
s3 = boto3.client('s3')

# Cache local pour les modèles chargé en mémoire de la Lambda (Warm Start)
_MODELS = {}

def load_model_from_s3(key):
    if key in _MODELS:
        return _MODELS[key]
    
    print(f"📥 Téléchargement du modèle {key} depuis S3...")
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    model_data = response['Body'].read()
    model = joblib.load(io.BytesIO(model_data))
    
    _MODELS[key] = model
    return model

def lambda_handler(event, context):
    try:
        # 1. Parsing Input
        body = json.loads(event.get('body', '{}'))
        pathology = event.get('queryStringParameters', {}).get('type', 'cancer')
        
        # 2. Chargement du modèle/scaler approprié
        if pathology == 'cancer':
            model = load_model_from_s3('models/breast_cancer_best_model.joblib')
            scaler = load_model_from_s3('models/breast_cancer_scaler.joblib')
        else:
            model = load_model_from_s3('models/heart_disease_best_model.joblib')
            scaler = load_model_from_s3('models/heart_disease_scaler.joblib')
            
        # 3. Préparation & Inférence
        input_df = pd.DataFrame([body])
        if pathology == 'cancer':
            input_df.columns = [c.replace('_', ' ') if 'concave' in c else c for c in input_df.columns]
            
        scaled_data = scaler.transform(input_df)
        proba = model.predict_proba(scaled_data)[0][1]
        prediction = int(model.predict(scaled_data)[0])
        
        # 4. Réponse
        result = {
            "pathology": pathology,
            "prediction": str(prediction),
            "probability": round(float(proba), 4),
            "status": "success"
        }
        
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
