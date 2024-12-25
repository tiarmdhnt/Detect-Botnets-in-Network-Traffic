# Detect-Botnets-in-Network-Traffic
Application of Deep Learning to Detect Botnets in Network Traffic Using CTU-13 Dataset

This repository contains code and resources for building a machine learning-based botnet detection system using network traffic data. The project leverages Python libraries for data preprocessing, visualization, and model building.

# **Features**

- Data Preprocessing: Handle missing values, feature scaling, and label encoding.
- Data Visualization: Analyze label distributions and other key features using Seaborn and Matplotlib.
- Deep Learning Model: Implementation of a deep learning model using PyTorch for botnet detection.

# **Dataset**
The dataset used in this project is publicly available and can be downloaded directly:
- Source: CTU-Malware-Capture-Botnet-42
- Download Command:
  ```bash
  !wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow
  
# **Installation**
To use this project, ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn torch
```
# **Project Workflow**
**1. Data Preprocessing**
- Load the dataset using Pandas.
- Handle missing values.
- Normalize numerical features using StandardScaler.Encode categorical labels using LabelEncoder.

**2. Data Visualization**
- Plot label distributions using Seaborn and Matplotlib to understand the data.

**3. Machine Learning Model**
- Model Architecture:
- Input layer for network traffic features.
- Two hidden layers with ReLU activation.
- Output layer with softmax for multi-class classification.
- Framework: PyTorch.

**4. Training and Evaluation**
- Split dataset into training and testing sets.
- Train the model using the Adam optimizer and CrossEntropyLoss.
- Evaluate the model using metrics like accuracy, precision, recall, and F1-score.

# **Code Snippets**
**Data Loading and Preprocessing**
```bash
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = "/content/capture20110810.binetflow"
data = pd.read_csv(file_path, delimiter=',')
data = data.dropna()

# Feature scaling
scaler = StandardScaler()
data[['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']] = scaler.fit_transform(data[['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']])

# Label encoding
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])
```
**Model Definition**
```bash
import torch
import torch.nn as nn

class BotnetDetectionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BotnetDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
**Training Loop**
```bash
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# Data preparation
X = data.drop(columns=['Label'])
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model initialization
model = BotnetDetectionModel(input_dim=X_train.shape[1], num_classes=len(y.unique()))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
```bash
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
```

# **Metrics**
The model evaluates the following metrics during training and testing:
- Accuracy
- Precision
- Recall
- F1-Score

# **Technology Used**
- Python
- PyTorch
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
