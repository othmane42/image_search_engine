# Spécifier une version python à utiliser
FROM python:3.7-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le contenu du répertoire courant dans le conteneur à /app
COPY . .

# Installer les packages spécifiés dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mettre à jour apt-get et installer ffmpeg, libsm6 et libxext6
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Rendre le port 5000 accessible au monde extérieur à ce conteneur
EXPOSE 5000

# Définir la variable d'environnement
ENV FLASK_APP=main.py

# Exécuter app.py lorsque le conteneur se lance
CMD ["flask", "run", "--host=0.0.0.0"]
