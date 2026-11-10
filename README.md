# Étude d’attaques adversariales sur des détecteurs YOLO

Ce projet expérimental compare trois méthodes d’optimisation d’images adversariales contre des détecteurs d’objets YOLO. Les scripts cherchent à réduire les scores associés à la classe **« person »** sur plusieurs modèles, puis évaluent les détections obtenues.

> **Usage responsable :** ce dépôt est destiné à l’apprentissage et à la recherche sur la robustesse des modèles, dans un environnement contrôlé et avec des images et modèles que vous êtes autorisé à utiliser. Il ne faut pas utiliser ces méthodes pour contourner des dispositifs de sécurité ou de surveillance. Les résultats dépendent des modèles, des images et de l’environnement d’exécution ; aucune réussite n’est garantie.

## Méthodes étudiées

Les trois scripts travaillent sur une image normalisée et limitent la perturbation pixel par pixel avec une contrainte de norme **L∞**. L’objectif commun est de modifier les scores de détection de la classe « person » sur un ensemble de modèles.

| Script | Méthode | Principe |
|---|---|---|
| [`CW.py`](CW.py) | Carlini & Wagner (C&W), adaptée avec une contrainte L∞ | Optimise une perturbation à l’aide d’Adam. Une transformation `tanh` maintient la perturbation dans les limites définies ; la fonction de coût combine l’objectif de réduction des scores de détection et une pénalité de distorsion. |
| [`PGD.py`](PGD.py) | Projected Gradient Descent (PGD) | Effectue des mises à jour itératives dans la direction du gradient, puis reprojette l’image dans la région autorisée autour de l’image d’origine. Le script initialise aussi la perturbation de façon aléatoire dans cette région. |
| [`MI_FGSM.py`](MI_FGSM.py) | Momentum Iterative FGSM (MI-FGSM) | Utilise des mises à jour itératives par signe du gradient et accumule une direction de momentum. Les modèles sont traités séquentiellement afin de réduire l’occupation mémoire. |

Les scripts PGD et C&W chargent un ensemble de modèles YOLO en mémoire pour calculer conjointement l’objectif. MI-FGSM recharge les modèles de l’ensemble au fil des itérations. Les scripts tentent ensuite d’évaluer les détections sur les différents modèles.

## Organisation des fichiers

- `CW.py` : pipeline C&W et fonctions d’évaluation/visualisation.
- `PGD.py` : pipeline PGD en ensemble.
- `MI_FGSM.py` : pipeline MI-FGSM en ensemble.
- `Adversarial_Methods.pdf` : document de référence fourni avec le projet.
- Les fichiers PNG présents à la racine sont des exemples d’images associés aux expérimentations.

## Fonctionnement actuel et limites

Les scripts sont des prototypes autonomes, et non une bibliothèque avec une interface en ligne de commande. Les chemins d’entrée et de sortie sont définis directement dans le code (actuellement sous `/content/`, configuration typique de Google Colab) et doivent être adaptés à l’environnement d’exécution. Les images sont redimensionnées en 640 × 640 ; les résultats générés sont enregistrés en PNG.

Les noms des poids YOLO sont également définis dans les scripts. Leur disponibilité dépend de l’environnement et de la version de la bibliothèque Ultralytics. Les dépendances importées comprennent PyTorch, Ultralytics, NumPy, OpenCV, Pillow et Matplotlib. Aucun fichier de dépendances avec versions verrouillées n’est fourni pour le moment ; les résultats peuvent donc varier selon les versions installées, les poids disponibles et le matériel.

L’accélération CUDA est utilisée si elle est disponible ; sinon, les scripts sélectionnent le CPU. L’utilisation mémoire et le temps de calcul peuvent être importants, notamment lorsque plusieurs modèles sont chargés simultanément. Le décodage AVIF dépend de la prise en charge du format par Pillow dans l’environnement installé.

## Cadre expérimental recommandé

Pour interpréter les résultats, consignez au minimum l’image et son origine autorisée, les modèles et leurs versions, les paramètres de l’expérience, le matériel utilisé ainsi que les détections avant et après l’expérience. Une image visuellement similaire ne garantit pas une détection identique par d’autres modèles ou dans d’autres conditions.
