import json
import boto3
import os

sns = boto3.client('sns')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')

def lambda_handler(event, context):
    print("🔔 S3 Event received:", json.dumps(event))
    
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        event_name = record['eventName']
        
        # Logique de rotation ou alerte
        if 'models/' in key:
            message = f"DÉPÔT DE MODÈLE DÉTECTÉ : Le fichier {key} a été ajouté/modifié via {event_name}. La rotation automatique ou la mise à jour de l'API EC2 est requise."
            
            print(f"📢 Notification: {message}")
            if SNS_TOPIC_ARN:
                sns.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Message=message,
                    Subject="MediScan - Alerte Rotation Modèle"
                )
        
    return {
        "statusCode": 200,
        "body": json.dumps("Rotation/Notification terminée")
    }
