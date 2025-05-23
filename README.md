# Système de Question-Réponse sur Documents PDF avec RAG et Ollama

## Description
Ce projet implémente un système de question-réponse privé et sécurisé basé sur la méthode Retrieval-Augmented Generation (RAG).  
Il permet de poser des questions à partir du contenu de documents PDF, en exploitant un modèle de langage LLaMA via Ollama, tout en garantissant la confidentialité des données grâce à un traitement local.


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

## Setup supplémentaire

Avant de lancer l'application, merci de créer manuellement les dossiers et fichiers suivants, qui sont exclus du contrôle de version via `.gitignore` :

- `temp_uploads/` : dossier temporaire pour stocker les fichiers PDF téléchargés.
- `uploads/` : dossier pour stocker les fichiers PDF persistants.
- `api_key.txt` : fichier pour y placer vos clés API ou secrets (à ne pas partager).

Pour créer ces dossiers et fichiers sous Linux/macOS ou Windows (PowerShell), vous pouvez exécuter :

```bash
mkdir temp_uploads uploads
type nul > api_key.txt  # Windows
# ou
touch api_key.txt       # Linux/macOS
```

Assurez-vous d'ajouter vos clés ou secrets dans `api_key.txt` avant d'utiliser l'application.


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


