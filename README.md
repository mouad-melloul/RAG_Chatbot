# Système de Question-Réponse sur Documents PDF avec RAG et Ollama

## Description
Ce projet implémente un système de question-réponse privé et sécurisé basé sur la méthode Retrieval-Augmented Generation (RAG).  
Il permet de poser des questions à partir du contenu de documents PDF, en exploitant un modèle de langage LLaMA via Ollama, tout en garantissant la confidentialité des données grâce à un traitement local.

## Démarrage du serveur Ollama

Avant de lancer l’application Flask, il faut démarrer le serveur Ollama localement, qui écoute sur le port 11434.

1. Ouvrez un terminal et lancez la commande suivante :

```
ollama serve
```
## Installation

1. Cloner le dépôt :
```
git clone <URL_DU_DEPOT>
cd <NOM_DU_PROJET>
```
2. Créer un environnement virtuel (recommandé) :
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Installer les dépendances :
```
pip install -r requirements.txt
```
4. Installer le modèle LLaMA recommandé avec Ollama :
```
ollama pull llama3.2:1b
```

## Setup supplémentaire

Avant de lancer l'application, merci de créer manuellement le dossier suivant :

- `temp_uploads/` : dossier temporaire pour stocker les fichiers PDF téléchargés.

## Utilisation

1. Lancer l’application Flask :  
```
python app.py
```
2. Ouvrir votre navigateur et aller à l’adresse : 
 http://127.0.0.1:5000/
3. Télécharger un fichier PDF via l’interface web.
4. Poser vos questions sur le contenu du PDF.
5. Recevoir des réponses contextuelles générées localement par le modèle LLaMA via Ollama.


