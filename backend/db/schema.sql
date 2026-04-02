-- Schéma SQL pour MediScan Africa (RDS Aurora)

CREATE DATABASE IF NOT EXISTS mediscan;
USE mediscan;

-- 1. Table des Patients
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_uuid VARCHAR(50) UNIQUE NOT NULL,
    age INT,
    sexe VARCHAR(1), -- 'M' ou 'F'
    region VARCHAR(50), -- Exemple: 'Dakar', 'Thies'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Table des Diagnostics (Historique)
CREATE TABLE IF NOT EXISTS diagnostics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    pathology VARCHAR(50), -- 'Cancer' ou 'Coeur'
    prediction VARCHAR(50),
    probability FLOAT,
    severity VARCHAR(20),
    recommendations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- 3. Table des Métriques Modèles (Audit)
CREATE TABLE IF NOT EXISTS model_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(50),
    version VARCHAR(20),
    auc_roc FLOAT,
    sensitivity FLOAT,
    accuracy FLOAT,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
