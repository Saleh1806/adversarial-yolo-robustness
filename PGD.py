"""
=========================================================================================
SCRIPT PRINCIPAL : Attaque Adversariale d'Évasion PGD par Ensemble YOLO 
=========================================================================================
Ce script génère des perturbations contradictoires furtives et optimisées pour supprimer 
la détection de personnes sur des images cibles en utilisant une approche d'ensemble pure 
sous PyTorch (combinant 10 architectures de détection YOLO de la v5 à la v26).

Caractéristiques principales :
- Attaque PGD (Projected Gradient Descent) conjointe sur un ensemble de modèles chargés en VRAM.
- Initialisation aléatoire uniforme dans la boule epsilon L-infini pour éviter les minima locaux.
- Contrainte stricte de norme L-infini (eps=8/255) pour garantir l'invisibilité à l'œil nu.
- Chargement robuste des images (support natif et sans perte des formats .avif via Pillow).
- Sauvegarde finale rigoureuse au format sans perte (.png) pour préserver le bruit contradictoire.
- Évaluation exhaustive modèle par modèle avec affichage des résultats et visualisations.
"""

import os
import time
import gc
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# Option PyTorch pour éviter la fragmentation de la mémoire CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def ensemble_pgd_attack_pure(
    models_list,
    image_tensor,
    eps=8/255,
    alpha=1.5/255,
    num_iter=60
):
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Exécute l'attaque itérative PGD (Projected Gradient Descent) avec une contrainte 
    de norme L-infini stricte (eps=8/255) sur un ensemble de modèles YOLO chargés en VRAM. 
    Elle cumule les pertes et rétropropage les gradients de tous les modèles à chaque itération.
    
    ENTRÉES :
        - models_list (list[YOLO]) : Liste des objets modèles YOLO initialisés et placés sur le GPU.
        - image_tensor (torch.Tensor) : Tenseur de l'image d'origine normalisé [0, 1] de forme [1, C, H, W].
        - eps (float) : Budget maximal de perturbation L-infini (fixé à 8/255 pour la furtivité).
        - alpha (float) : Pas de déplacement (taux d'apprentissage) par itération (défini à 1.5/255).
        - num_iter (int) : Nombre total d'itérations d'optimisation (par défaut 60).
        
    SORTIE :
        - x_adv (torch.Tensor) : Le tenseur de l'image finale corrompue (adversariale) après projection.
    ===========================================================================
    """
    device = image_tensor.device
    x_orig = image_tensor.clone().detach()

    # Initialisation avec un bruit aléatoire uniforme dans la boule epsilon (norme L-inf)
    x_adv = x_orig.clone().detach() + torch.FloatTensor(*x_orig.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    person_class_idx = 0  # Index de la classe 'person' dans le dataset COCO

    for iteration in range(num_iter):
        x_adv.requires_grad_(True)
        total_loss = 0.0

        # Calcul cumulé des pertes sur tous les modèles de l'ensemble
        for model in models_list:
            model.model.to(device)
            model.model.eval()

            outputs = model.model(x_adv)
            try:
                pred = outputs[0]
                person_scores = pred[:, 4 + person_class_idx, :]
                loss = person_scores.mean()
            except Exception:
                loss = outputs[0].mean()

            total_loss += loss

        # Rétropropagation conjointe de l'ensemble
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        total_loss.backward()

        grad = x_adv.grad
        if grad is None:
            break

        # Descente de gradient signée (méthode PGD standard)
        x_adv = x_adv.detach() - alpha * grad.sign()

        # Projection stricte dans la boule L-infini [-eps, eps] et respect des bornes de pixels [0.0, 1.0]
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, min=0.0, max=1.0).detach()

        # Nettoyage VRAM intermédiaire
        torch.cuda.empty_cache()

    return x_adv

def process_pure_ensemble_pipeline():
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Coordonne l'ensemble du pipeline d'exécution : chargement des 10 modèles YOLO en VRAM, 
    chargement sécurisé des images cibles, génération 
    de l'attaque PGD conjointe, sauvegarde sans perte au format .png, et évaluation 
    exhaustive modèle par modèle avec affichage graphique.
    
    ENTRÉES :
        - Aucune (utilise les chemins de fichiers et listes de modèles configurés en interne).
        
    SORTIE :
        - Aucune (écrit les images sur le disque dans le dossier de sortie et affiche les graphiques).
    ===========================================================================
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Utilisation de :', device)

    # Liste complète des 10 modèles pour l'ensemble
    ensemble_model_names = [
        'yolov5n.pt',
        'yolov5x.pt',
        'yolov8n.pt',
        'yolov8x.pt',
        'yolo11n.pt',
        'yolo11x.pt',
        'yolo12n.pt',
        'yolo12x.pt',
        'yolo26n.pt',
        'yolo26x.pt'
    ]

    print("Chargement de l'ensemble complet de modèles en mémoire...")
    models_list = []
    for name in ensemble_model_names:
        try:
            m = YOLO(name)
            m.model.to(device)
            models_list.append(m)
        except Exception as e:
            print(f"Erreur chargement {name}: {e}")

    # Liste des images cibles à traiter
    input_paths = [
        '/content/4.jpg'
    ]

    output_dir = '/content/adversarial_images'
    os.makedirs(output_dir, exist_ok=True)

    for img_path in input_paths:
        if not os.path.exists(img_path):
            print(f'Fichier non trouvé : {img_path}')
            continue

        print(f'\n================ TRAITEMENT ENSEMBLE PUR DE : {os.path.basename(img_path)} ================')

        # Chargement robuste gérant les .avif récalcitrants sans perte de données via Pillow
        try:
            if img_path.lower().endswith('.avif'):
                pil_img = Image.open(img_path).convert('RGB')
                img = np.array(pil_img)
            else:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f'Échec de lecture OpenCV pour : {img_path}')
                    continue
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Erreur critique lors du décodage de {img_path} : {e}")
            continue

        img_resized = cv2.resize(img, (640, 640))

        # Préparation du tenseur d'entrée normalisé [0, 1] et mis au format [B, C, H, W]
        x = np.transpose(img_resized.astype(np.float32) / 255.0, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        image_tensor = torch.tensor(x, dtype=torch.float32, device=device)

        # Lancement de l'attaque par ensemble PGD pure PyTorch avec contrainte L-inf de 8/255
        print("Lancement de l'attaque conjointe (Ensemble PGD PyTorch Pur L-inf)...")
        image_adv_tensor = ensemble_pgd_attack_pure(
            models_list=models_list,
            image_tensor=image_tensor,
            eps=8/255,
            alpha=1.5/255,
            num_iter=60
        )

        # Conversion finale du tenseur adversarial en tableau d'images numpy uint8
        img_adv = image_adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_adv = np.clip(img_adv * 255, 0, 255).astype(np.uint8)

        # Forçage de l'extension .png (lossless) pour préserver les perturbations contradictoires
        filename, _ = os.path.splitext(os.path.basename(img_path))
        save_filename = f'pure_ensemble_adv_{filename}.png'
        save_path = os.path.join(output_dir, save_filename)

        # Sauvegarde au format PNG via OpenCV (nécessite la conversion RGB vers BGR)
        cv2.imwrite(save_path, cv2.cvtColor(img_adv, cv2.COLOR_RGB2BGR))
        print(f'Image finale sauvegardée (sans perte .png) sous : {save_path}')

        # Évaluation exhaustive sur TOUS les 10 modèles un par un
        print("\n--- Évaluation finale sur l'ensemble des 10 modèles ---")
        for m_name in ensemble_model_names:
            try:
                eval_model = YOLO(m_name)
                eval_model.model.to(device)
                results = eval_model.predict(source=img_adv, conf=0.25, verbose=False)[0]

                num_persons = sum(1 for box in results.boxes if results.names[int(box.cls.cpu().numpy()[0])] == "person")
                print(f"Modèle {m_name:<12} -> Personnes détectées : {num_persons}")

                # Affichage graphique des résultats d'inférence pour chaque modèle
                plt.figure(figsize=(6, 4))
                plt.imshow(results.plot()[..., ::-1])
                plt.title(f'Eval {m_name} - {save_filename}')
                plt.axis('off')
                plt.show()

                del eval_model
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Erreur d'évaluation avec {m_name} : {e}")

    # Nettoyage final des modèles de l'ensemble de la mémoire GPU
    for m in models_list:
        del m
    torch.cuda.empty_cache()
    gc.collect()
    print("\nPipeline d'ensemble terminé avec succès !")

if __name__ == "__main__":
    process_pure_ensemble_pipeline()