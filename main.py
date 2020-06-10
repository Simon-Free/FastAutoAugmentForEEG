if __name__ == "main":
    # Appelle les données à l'aide de data_loader.
    # Data_loader construit les epochs à l'aide de epoch_constructor
    # Ensuite, Enrich_data sera la classe principale permettant d'enrichir les données (code de FastAutoAugment adapté)
    # (j'aime bien les classes, parce qu'elles permettent de partager les attributs
    # sans passer d'arguments params à chaque fois dans les fonctions)
    # elle appellerait un ensemble de sous .py qui permettraient de faire tourner tout ça
    # Enfin, on aurait la définition et l'entraînement du modèle à proprement parler : définition de fonctions
    # d'entraînement, de test, de plot des modèles.
    # A côté de ça, on aurait des fonctions de sauvegarde et d'archivage des différentes expérimentations.