"""
=========================================================================================
SCRIPT PRINCIPAL : Attaque Adversariale d'Évasion par Ensemble (YOLO Only) sous PyTorch Pur
=========================================================================================
Ce script génère des perturbations contradictoires universelles et furtives pour supprimer 
la détection de personnes sur des images cibles (incluant la gestion native des formats .avif).

Caractéristiques principales :
- Ensemble pur YOLO (sans Transformers) pour une compatibilité de gradient homogène.
- Chargement séquentiel des modèles à chaque itération pour optimiser et économiser la VRAM.
- Attaque MI-FGSM (Momentum Iterative Fast Gradient Sign Method).
- Contrainte stricte de norme L-infini (Linf) avec un budget d'epsilon serré (eps=8/255) 
  pour garantir un rendu visuel totalement naturel et imperceptible à l'œil nu.
- Sauvegarde finale rigoureuse au format sans perte (.png) pour préserver le bruit contradictoire.
- Évaluation exhaustive modèle par modèle avec affichage des résultats.
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


def ensemble_linf_mifgsm_attack(
    model_names,
    image_tensor,
    eps=8/255,
    num_iter=60,
    mu=1.0
):
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Exécute l'attaque itérative MI-FGSM avec une contrainte de norme L-infini (Linf) 
    stricte sur un ensemble de modèles YOLO chargés séquentiellement pour préserver la VRAM.
    
    ENTRÉES :
        - model_names (list[str]) : Liste des noms de fichiers de poids des modèles YOLO.
        - image_tensor (torch.Tensor) : Tenseur de l'image d'origine normalisé [0, 1] de forme [1, C, H, W].
        - eps (float) : Budget maximal de perturbation L-infini (défini par défaut à 8/255).
        - num_iter (int) : Nombre d'itérations d'optimisation de l'attaque (par défaut 60).
        - mu (float) : Facteur de momentum pour stabiliser la descente de gradient (par défaut 1.0).
        
    SORTIE :
        - x_adv (torch.Tensor) : Le tenseur de l'image finale corrompue (adversariale) après projection.
    ===========================================================================
    """
    device = image_tensor.device
    x_orig = image_tensor.clone().detach()
    x_adv = x_orig.clone().detach()

    alpha = eps / num_iter * 1.5
    g = torch.zeros_like(x_orig)
    person_class_idx = 0  # Index de la classe 'person' dans le dataset COCO

    for iteration in range(num_iter):
        x_adv.requires_grad_(True)
        total_loss = 0.0

        # Chargement séquentiel et libération modèle par modèle pour optimiser l'empreinte VRAM
        for name in model_names:
            try:
                torch.cuda.empty_cache()
                gc.collect()

                model = YOLO(name)
                model.model.to(device)
                model.model.eval()

                # Inférence et calcul de la perte ciblée sur les scores de la classe 'person'
                outputs = model.model(x_adv)
                pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                person_scores = pred[:, 4 + person_class_idx, :]
                loss = person_scores.mean()
                total_loss += loss

                # Suppression immédiate du modèle de la mémoire GPU
                del model
            except Exception:
                continue

        # Rétropropagation combinée de l'ensemble
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        total_loss.backward()

        grad = x_adv.grad.data
        if grad is None:
            break

        # Normalisation L1 du gradient pour stabiliser la direction de mise à jour
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

        # Mise à jour du vecteur de Momentum
        g = mu * g + grad

        # Descente de gradient signée
        x_adv = x_adv.detach() - alpha * g.sign()

        # Projection stricte dans la boule L-infini [-eps, eps] et bornes de pixels [0.0, 1.0]
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, min=0.0, max=1.0).detach()

        # Nettoyage intermédiaire de la VRAM
        torch.cuda.empty_cache()
        gc.collect()

    return x_adv


def process_comprehensive_ensemble_pipeline():
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Coordonne le pipeline global : initialisation de l'environnement, chargement robuste 
    des images , exécution de l'attaque par ensemble, 
    sauvegarde sans perte au format PNG, et évaluation exhaustive finale de chaque modèle.
    
    ENTRÉES :
        - Aucune (utilise les chemins et listes de modèles configurés en interne).
        
    SORTIE :
        - Aucune (sauvegarde les images générées sur le disque et affiche les graphiques d'évaluation).
    ===========================================================================
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Utilisation de :', device)

    # Purge initiale complète de la VRAM
    print("Purge initiale de la mémoire GPU (VRAM)... ")
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.1)

    # Liste exclusive des modèles de détection YOLO composant l'ensemble
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

        print(f'\n================ TRAITEMENT L-INF (ENSEMBLE YOLO) DE : {os.path.basename(img_path)} ================')

        # Chargement robuste gérant les .avif récalcitrants sans perte de précision
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

        # Conversion et préparation du tenseur d'entrée normalisé [0, 1]
        x = np.transpose(img_resized.astype(np.float32) / 255.0, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        image_tensor = torch.tensor(x, dtype=torch.float32, device=device)

        # Lancement de l'attaque par ensemble conjointe (L-inf / MI-FGSM)
        print("Lancement de l'attaque conjointe Linf (Ensemble YOLO, eps=8/255)...")
        image_adv_tensor = ensemble_linf_mifgsm_attack(
            model_names=ensemble_model_names,
            image_tensor=image_tensor,
            eps=8/255,          # Budget L-infini strict assurant la discrétion visuelle
            num_iter=40,
            mu=1.0
        )

        # Conversion finale du tenseur adversarial en tableau d'images numpy uint8
        img_adv = image_adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_adv = np.clip(img_adv * 255, 0, 255).astype(np.uint8)

        # Sauvegarde stricte sans perte au format .png pour interdire l'altération du bruit
        filename, _ = os.path.splitext(os.path.basename(img_path))
        save_filename = f'pure_ensemble_linf_{filename}.png'
        save_path = os.path.join(output_dir, save_filename)

        cv2.imwrite(save_path, cv2.cvtColor(img_adv, cv2.COLOR_RGB2BGR))
        print(f'Image finale sauvegardée sans perte (.png) sous : {save_path}')

        # Évaluation exhaustive sur l'ensemble des modèles YOLO un par un
        print("\n--- Évaluation finale exhaustive sur l'ensemble des modèles YOLO ---")
        for m_name in ensemble_model_names:
            try:
                eval_model = YOLO(m_name)
                eval_model.model.to(device)
                results = eval_model.predict(source=img_adv, conf=0.25, verbose=False)[0]

                num_persons = sum(1 for box in results.boxes if results.names[int(box.cls.cpu().numpy()[0])] == "person")
                print(f"Modèle {m_name:<12} -> Personnes détectées : {num_persons}")

                # Affichage graphique des boîtes de détection restantes (le cas échéant)
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

    # Nettoyage final de la VRAM
    torch.cuda.empty_cache()
    gc.collect()
    print("\nPipeline complet (Linf + Ensemble YOLO pur) exécuté avec succès !")


if __name__ == "__main__":
    process_comprehensive_ensemble_pipeline()