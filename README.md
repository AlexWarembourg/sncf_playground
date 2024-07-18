# Projet bac à sable de Prédiction de l'affluence 

## Description
Ce projet vise à prédire l'affluence dans plus de 400 stations de gare pour les six prochains mois en utilisant des techniques de machine learning.

## Structure du Répertoire
- `data/`: Contient les données brutes et transformées.
- `notebooks/`: Notebooks Jupyter pour l'exploration et la modélisation.
- `out/models`: Modèle entraîné et informations associées.
- `src/`: Scripts de préparation des données, entraînement du modèle et prédiction, ainsi que les fichiers de l'API.
- `streamlit_app/`: Application Streamlit pour l'interface utilisateur.

## Syncing Dependencies

This project uses Poetry for dependency management. To ensure that `requirements.txt` is always up-to-date with `pyproject.toml`, we use a synchronization script and a GitHub Actions workflow.

### Running the Sync Script Locally

If you update `pyproject.toml`, run the following command to sync dependencies:

```sh
./sync_dependencies.sh
