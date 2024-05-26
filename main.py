import os
import operator
import csv
import pickle
from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from scipy.spatial.distance import correlation
import tensorflow as tf
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import cosine_distances
from tensorflow.keras.applications import vgg16, resnet50, mobilenet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

import matplotlib.pyplot as plt
import math
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load pickled descriptor data
descriptors_data = {
    'vgg16': 'static/features/VGG16.pkl',
    'resnet50': 'static/features/Resnet50.pkl',
    'mobilenet':'static/features/MobileNet.pkl'
}

preprocess_fun = {
    'vgg16':vgg16_preprocess,
    'mobilenet':mobilenet_preprocess,
    'resnet50':resnet50_preprocess,
}


features_data = {key: pickle.load(open(path, 'rb')) for key, path in descriptors_data.items()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img_path, model_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_fun[model_name](img_data)
    if model_name == 'vgg16':
        model = vgg16.VGG16(weights='imagenet', include_top=True, pooling='avg')
    elif model_name == 'resnet50':
        model = resnet50.ResNet50(weights='imagenet',include_top=True, pooling='avg')
    elif model_name =="mobilenet":
        model = mobilenet.MobileNet(weights='imagenet', include_top=True, pooling='avg')
  
    features = model.predict(img_data)
    return features.flatten()

def euclidianDistance(l1,l2):
    distance = 0
    length = min(len(l1),len(l2))
    for i in range(length):
        distance += pow((l1[i] - l2[i]), 2)
    return math.sqrt(distance)

def chiSquareDistance(l1, l2):
    n = min(len(l1), len(l2))
    return np.sum((l1[:n] - l2[:n])**2 / l2[:n])

def bhatta(l1, l2):
    n = min(len(l1), len(l2))
    N_1, N_2 = np.sum(l1[:n])/n, np.sum(l2[:n])/n
    score = np.sum(np.sqrt(l1[:n] * l2[:n]))
    num = round(score, 2)
    den = round(math.sqrt(N_1*N_2*n*n), 2)
    return math.sqrt( 1 - num / den )

def flann(a,b):
  a = np.float32(a)
  b = np.float32(b)
  FLANN_INDEX_KDTREE = 1
  INDEX_PARAM_TREES = 5
  SCH_PARAM_CHECKS = 50
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
  sch_params = dict(checks=SCH_PARAM_CHECKS)
  flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
  matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
  return np.mean(matches)

def bruteForceMatching(a, b):
    a = a.astype('uint8')
    b = b.astype('uint8')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)



def getkVoisins(lfeatures, test, k,metric, image_class=None):
    ldistances = []
    for i in range(len(lfeatures)):
        if metric=="euclidean":
            dist = euclidianDistance(test[1], lfeatures[i][1])
        elif metric=="flann":
            dist = flann(test[1], lfeatures[i][1])
        elif metric=="brute-force-matcher":
             dist = bruteForceMatching(test[1], lfeatures[i][1])
        elif metric=="bhattacharyya":
             dist = bhatta(test[1], lfeatures[i][1])
        elif metric=="chi-square":
              dist = chiSquareDistance(test[1], lfeatures[i][1])
        elif metric=="correlation":
              dist = correlation(test[1], lfeatures[i][1])  
        elif metric=="cosine_distance":
              dist = cosine_distances(test[1].reshape(-1,1), lfeatures[i][1].reshape(-1,1))  

        
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist))
    ldistances.sort(key=operator.itemgetter(2))
    return ldistances[:k]

def recherche(image_req, descriptor,metric, top=20, image_class=None):
    
    features=features_data[descriptor]
    if image_class!=None:
        features=filter(lambda x: (int(os.path.splitext(os.path.basename(x[0]))[0])//100)==image_class,features)
        features=list(features)
    voisins = getkVoisins(features, image_req, 
                          top,metric, image_class)
    nom_images_proches = []
    for k in range(len(voisins)):
        img_path = os.path.join("static", voisins[k][0])
        nom_images_proches.append(img_path)
    return nom_images_proches

def Compute_RP(RP_file, top, nom_image_requete, nom_images_non_proches):
    rappel_precision = []
    rp = []
    position1 = int(nom_image_requete) // 100
    for j in range(top):
        id_img = os.path.splitext(os.path.basename(nom_images_non_proches[j]))[0]
        position2 = int(id_img) // 100
        if position1 == position2:
            rappel_precision.append("pertinant")
        else:
            rappel_precision.append("non pertinant")

    for i in range(top):
        j = i
        val = 0
        while j >= 0:
            if rappel_precision[j] == "pertinant":
                val += 1
            j -= 1
        rp.append((val / (i + 1)) * 100)
        rp.append((val / top) * 100)
    with open(RP_file, 'w') as s:
        writer = csv.writer(s, delimiter=' ')
        for i in range(0, len(rp), 2):
            writer.writerow([rp[i], rp[i+1]])

    # Print RP file contents for debugging
    with open(RP_file, 'r') as s:
        content = s.read()

def Display_RP(fichier,descriptor):
    x = []
    y = []
    with open(fichier) as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for row in plots:
            if len(row) == 2:
                x.append(float(row[0]))
                y.append(float(row[1]))
    plt.figure()
    plt.plot(y, x, 'C1', label=descriptor)
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title("R/P")
    plt.legend()
    plt.savefig('static/uploads/precision_recall_curve.png')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"Uploaded file saved to {file_path}")
        return jsonify({'filename': filename, 'file_path': file_path})
    return jsonify({'error': 'Invalid file format'})

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file {file_path}")
        return jsonify({'message': 'File deleted'})
    return jsonify({'error': 'File not found'})

@app.route('/search', methods=['POST'])
def search():
    # Récupération des données du formulaire html
    filename = request.form.get('filename')
    descriptor = request.form.get('descriptor')
    metric = request.form.get('similarity')
    image_class = request.form.get('image_class')
    topn = int(request.form.get('topn'))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        print("File not found error")
        return jsonify({'error': 'File not found'})
    # Extraction des caractérisques de l'image requête
    image_features = extract_features(file_path, descriptor)
    image_class = None if image_class=="" else int(image_class)

    # Effectuer la recherche sur la base de données
    nom_images_proches = recherche((file_path, image_features), descriptor,
                                   metric, top=topn, image_class=image_class)

    # Calcul et affichage de la courbe r/p
    rp_file = 'static/uploads/rp_file.txt'
    nom_image_requete = os.path.splitext(os.path.basename(file_path))[0]
    Compute_RP(rp_file, topn, nom_image_requete, nom_images_proches)
    Display_RP(rp_file,descriptor)
    rp_curve = 'static/uploads/precision_recall_curve.png'
    # Afficher le résultat 
    return jsonify({
        'top20_similar_images': nom_images_proches,
        'rp_curve': rp_curve
    })

if __name__ == '__main__':
    app.run(debug=True)
