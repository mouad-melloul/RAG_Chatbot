# Système de Question-Réponse sur Documents PDF avec RAG et Ollama

## Description
Ce projet implémente un système de question-réponse privé et sécurisé basé sur la méthode Retrieval-Augmented Generation (RAG).  
Il permet de poser des questions à partir du contenu de documents PDF, en exploitant un modèle de langage LLaMA via Ollama, tout en garantissant la confidentialité des données grâce à un traitement local.

## Structure du projet
C:.
│ .env
│ api_key.txt
│ app.py                # Application Flask principale
│ requirements.txt      # Dépendances Python
│
├───notebooks
│ 3.chat_myPDF.ipynb    # Notebook d'expérimentation
│
├───static
│ │ script.js           # Scripts JavaScript front-end
│ │ style.css           # Feuilles de style CSS
│ │
│ └───images
│ Baymax.jpeg
│ PDFLogo.png
│ user.png
│
├───templates
│ chat.html              # Template HTML de l'interface utilisateur
│
├───temp_uploads         # Dossier pour fichiers PDF temporaires
│
└───uploads              # Dossier pour fichiers PDF permanents



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


