import os
from tkinter import Image
from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Prepare data

Img2Vec = Img2Vec()

data_dir = "./weather_dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_).convert('RGB')

            img_features = Img2Vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels


# train model
model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])


#evaluate model
y_predictions = model.predict(data['validation_data'])
accuracy = accuracy_score( y_predictions, data['validation_labels'])
report = classification_report(y_predictions, data['validation_labels'])

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

#save model 
with open("./random_forest_model.pkl", "wb") as f:
    pkl.dump(model, f)
    f.close()
