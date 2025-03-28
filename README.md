Aqui está uma versão aprimorada do seu README com melhorias na clareza, formatação e organização para facilitar o entendimento e a execução do projeto:

---

# Projeto: Servidor MLflow na AWS com Armazenamento S3 e Modelo ElasticNet

Este guia fornece um processo passo a passo para configurar o **MLflow** em uma instância EC2 da AWS, armazenando artefatos no S3. O projeto inclui a criação de um servidor MLflow, configuração de permissões IAM, e a implementação de um modelo ElasticNet para regressão.

---

## 📌 **Pré-requisitos**

Antes de começar, certifique-se de que você possui os seguintes itens configurados:

- Conta AWS ativa
- **AWS CLI** instalado e configurado
- Acesso ao terminal para executar comandos
- Permissões adequadas para criar e configurar recursos na AWS

---

## 🚀 **Passo a Passo para Configuração**

### 1️⃣ **Criar Usuário IAM**

1. Acesse o **AWS Console** e vá até **IAM**.
2. Crie um novo **usuário** com a permissão **AdministratorAccess**.
3. Salve a **Access Key ID** e **Secret Access Key**.
4. Configure suas credenciais na AWS CLI:
   ```bash
   aws configure
   ```

### 2️⃣ **Criar Bucket S3**

1. No **AWS Console**, vá para **S3** e crie um novo bucket para armazenar os artefatos do MLflow.
2. Anote o nome do bucket (exemplo: `mlflowtracking1`).

### 3️⃣ **Criar e Configurar Instância EC2**

1. No **AWS Console**, vá para **EC2** e inicie uma instância **Ubuntu**.
2. Configure um **Security Group** permitindo conexões na **porta 5000** (para o MLflow Server).
3. Conecte-se à instância via SSH.

### 4️⃣ **Instalar Dependências na EC2**

Execute os seguintes comandos para instalar as dependências:

```bash
# Atualizar pacotes
sudo apt update

# Instalar Python e gerenciadores de pacotes
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv

# Criar diretório para o MLflow
mkdir mlflow && cd mlflow

# Instalar dependências
pipenv install mlflow awscli boto3

# Ativar ambiente virtual
pipenv shell
```

### 5️⃣ **Configurar Credenciais AWS na EC2**

Dentro do ambiente virtual, configure as credenciais da AWS:

```bash
aws configure
```

### 6️⃣ **Iniciar o MLflow Server**

Inicie o servidor MLflow apontando para o bucket S3:

```bash
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowtracking1 --backend-store-uri sqlite:///mlflow.db
```

### 7️⃣ **Acessar o MLflow**

1. Abra o **IPv4 Público** da EC2 na porta **5000**.
2. No seu terminal local, configure a variável de ambiente `MLFLOW_TRACKING_URI` apontando para o servidor MLflow na EC2:

```bash
export MLFLOW_TRACKING_URI=http://<SEU_IP_PUBLICO>:5000/
```

---

## ✅ **Conclusão**

Agora o MLflow está configurado e rodando na AWS, com artefatos sendo armazenados no S3. 🎉

Para mais detalhes e ajuda, consulte a [documentação oficial do MLflow](https://mlflow.org/docs/latest/index.html).

---

## Visão Geral do Projeto

Este projeto configura um servidor **MLflow** em uma instância **EC2** (Ubuntu) na AWS. A configuração inclui permissões **IAM** para acesso seguro ao **S3** e o treinamento de um modelo de **ElasticNet**. O fluxo completo de treinamento do modelo é registrado e gerido no MLflow.

---

## Estrutura do Projeto

### 1. **Configuração da Instância EC2**

A instância EC2 é configurada para hospedar o servidor MLflow, incluindo:

- Instalação do MLflow e dependências.
- Configuração para comunicação remota com a instância.
- Criação de um bucket **S3** para armazenar artefatos de modelos.
- Definição de permissões **IAM** para acesso seguro ao S3.

### 2. **Instalação do MLflow na EC2**

Para instalar o MLflow na instância EC2, execute os seguintes comandos:

```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install mlflow boto3 scikit-learn pandas numpy
```

Inicie o servidor MLflow com o seguinte comando:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://<seu-bucket-s3>/ \
  --host 0.0.0.0 --port 5000
```

### 3. **Código Python para Treinamento e Registro no MLflow**

#### **3.1. Carregamento de Bibliotecas e Configuração**

Primeiro, importe as bibliotecas necessárias e configure o URI do MLflow:

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

os.environ["MLFLOW_TRACKING_URI"] = "http://<seu-endereco-ec2>:5000/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.amazonaws.com"
```

#### **3.2. Download e Processamento dos Dados**

O dataset `winequality-red.csv` é baixado e dividido em treino e teste:

```python
csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(csv_url, sep=";")
train, test = train_test_split(data)
X_train, X_test = train.drop(["quality"], axis=1), test.drop(["quality"], axis=1)
y_train, y_test = train[["quality"]], test[["quality"]]
```

#### **3.3. Definição e Treinamento do Modelo**

O modelo **ElasticNet** é treinado com os dados de treino:

```python
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(X_train, y_train)
```

#### **3.4. Avaliação do Modelo**

O modelo é avaliado utilizando **RMSE**, **MAE** e **R²**:

```python
predicted_qualities = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predicted_qualities))
mae = mean_absolute_error(y_test, predicted_qualities)
r2 = r2_score(y_test, predicted_qualities)
```

#### **3.5. Registro no MLflow**

Parâmetros e métricas do modelo são registrados no **MLflow**:

```python
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)

mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
```

### 4. **Integração com IAM e S3**

As permissões **IAM** são configuradas para garantir que o MLflow possa acessar e armazenar artefatos no S3 de maneira segura. A política IAM necessária é a seguinte:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::<seu-bucket-s3>/*"
    }
  ]
}
```

### 5. **Execução do Script**

Para rodar o script Python e treinar o modelo, use o seguinte comando:

```bash
python3 train.py 0.7 0.3
```

Isso treina um modelo **ElasticNet** com `alpha=0.7` e `l1_ratio=0.3`, e registra os resultados no **MLflow**.

---

## 🚀 **FINISH**

Agora você tem um servidor **MLflow** em execução na **AWS EC2**, com **S3** para armazenar artefatos e integração com **IAM** para garantir a segurança. O modelo **ElasticNet** é treinado e registrado de forma eficiente no **MLflow** para análise posterior.

Esse processo demonstra como configurar um ambiente de MLflow escalável e seguro na **AWS**.