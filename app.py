import os
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import logging

os.environ["MLFLOW_TRACKING_URI"] = "http://ec2-3-84-220-213.compute-1.amazonaws.com:5000/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.amazonaws.com"

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__=="__main__":
    # DATA INGESTION-READING THE DATASET -- WINE QUALITY DATASET
    csv_url=(
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
     
    try:
        data=pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.exception("Unable to download the data")
        
    # SPLIT THE DATA IN TRAIN AND TEST
    train,test=train_test_split(data)
    
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]
    
    # -------------------------------------------------------------------------------------------------------
    """
    # 📝 Documenting the model parameters 
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    
    Este bloco de código está configurando os valores dos hiperparâmetros alpha e l1_ratio para o modelo ElasticNet com base nos 
    argumentos fornecidos na linha de comando ao executar o script. Aqui está a explicação detalhada:
    
    1.  sys.argv: É uma lista que contém os argumentos passados para o script Python pela linha de comando. 
        O primeiro elemento (sys.argv[0]) é o nome do script, e os elementos subsequentes são os argumentos fornecidos.

    2.  len(sys.argv) > 1: Verifica se há pelo menos um argumento adicional fornecido além do nome do script. 
        Se sim, o valor do primeiro argumento (sys.argv[1]) será convertido para float e atribuído a alpha. Caso contrário,
        o valor padrão 0.5 será usado.

    3.  len(sys.argv) > 2: Verifica se há um segundo argumento adicional fornecido. 
        Se sim, o valor do segundo argumento (sys.argv[2]) será convertido para float e atribuído a l1_ratio. Caso contrário, 
        o valor padrão 0.5 será usado.
    """
    # -------------------------------------------------------------------------------------------------------
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    # -------------------------------------------------------------------------------------------------------
    
    with mlflow.start_run():
        lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(X_train,y_train)
        
        predicted_qualities = lr.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
        
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # 📌 FOR THE REMOTE SERVER AWS WE NEED TO DO THE SETUP
        remote_server_uri="http://ec2-3-84-220-213.compute-1.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(
                lr,"model",registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr,"model")
        