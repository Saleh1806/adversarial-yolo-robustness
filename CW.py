"""
=========================================================================================
SCRIPT PRINCIPAL : Attaque Adversariale d'Évasion C&W par Ensemble YOLO 
=========================================================================================
Ce script génère des perturbations contradictoires hautement optimisées et furtives pour 
supprimer la détection de personnes sur des images cibles en combinant une fonction de perte de type Carlini & Wagner (C&W) avec 
une contrainte stricte de norme L-infini (eps=32/255) sur un ensemble pur de 10 modèles YOLO.

Caractéristiques principales :
- Ensemble exclusif de 10 modèles YOLO (de la v5 à la v26, sans Transformers).
- Optimisation non contrainte via changement de variable (arctanh/tanh) pour respecter 
  strictement la boule L-infini et minimiser la distorsion.
- Chargement robuste des images et sauvegarde finale sans perte au format PNG.
- Évaluation exhaustive modèle par modèle avant et après l'attaque.
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

def extract_predictions(results):
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Extrait les classes, les boîtes de détection et les scores de confiance 
    à partir de l'objet de résultats brut renvoyé par l'inférence YOLO.
    
    ENTRÉES :
        - results (ultralytics.engine.results.Results) : Résultat d'inférence d'un modèle YOLO.
        
    SORTIES :
        - predictions_class (list[str]) : Noms des classes détectées.
        - predictions_boxes (list[list]) : Coordonnées des boîtes sous la forme [[(x1, y1), (x2, y2)], ...].
        - predictions_scores (list[float]) : Scores de confiance associés à chaque détection.
    ===========================================================================
    """
    predictions_class = []
    predictions_boxes = []
    predictions_scores = []

    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        score = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]

        predictions_class.append(results.names[cls_id])
        predictions_boxes.append([
            (float(x1), float(y1)),
            (float(x2), float(y2))
        ])
        predictions_scores.append(score)

    return predictions_class, predictions_boxes, predictions_scores

def plot_image_with_boxes(img, boxes, pred_cls):
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Dessine graphiquement les boîtes englobantes et les libellés de classes sur une 
    image, puis l'affiche à l'écran via Matplotlib.
    
    ENTRÉES :
        - img (np.ndarray) : Tableau numpy représentant l'image RGB de dimensions [H, W, 3].
        - boxes (list[list]) : Liste des coordonnées de boîtes de détection.
        - pred_cls (list[str]) : Liste des noms de classes correspondants.
        
    SORTIE :
        - Aucune (affiche directement la figure graphique).
    ===========================================================================
    """
    img = np.copy(img).astype(np.uint8)
    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            (int(boxes[i][1][0]), int(boxes[i][1][1])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            pred_cls[i],
            (int(boxes[i][0][0]), int(boxes[i][0][1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def ensemble_cw_linf_attack(
    models_list,
    image_tensor,
    eps=32/255,
    alpha=2e-2,
    num_iter=200,
    c=100
):
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Exécute l'attaque contradictoire par ensemble basée sur la formulation de Carlini & Wagner (C&W) 
    avec une contrainte de norme L-infini stricte, optimisant itérativement l'image pour 
    faire disparaître la détection de la classe 'person' sur l'ensemble des modèles YOLO.
    
    ENTRÉES :
        - models_list (list[YOLO]) : Liste des objets modèles YOLO chargés en mémoire GPU.
        - image_tensor (torch.Tensor) : Tenseur de l'image d'origine normalisé [0, 1] de forme [N, C, H, W].
        - eps (float) : Budget maximal de perturbation L-infini (fixé à 32/255).
        - alpha (float) : Taux d'apprentissage de l'optimiseur Adam (défini à 2e-2).
        - num_iter (int) : Nombre total d'itérations d'optimisation (par défaut 200).
        - c (float) : Facteur de pondération de la perte d'évasion par rapport à la distorsion (défini à 100).
        
    SORTIE :
        - x_adv (torch.Tensor) : Le tenseur de l'image finale adversarial après projection.
    ===========================================================================
    """
    device = image_tensor.device
    x_orig = image_tensor.clone().detach()
    
    # Initialisation avec un bruit aléatoire uniforme dans la boule L-inf pour éviter les minima locaux
    random_noise = torch.FloatTensor(*x_orig.shape).uniform_(-eps, eps).to(device)
    x_init = torch.clamp(x_orig + random_noise, 0.0, 1.0)
    
    # Changement de variable (C&W style) pour optimiser dans un domaine non contraint via arctanh
    w_init = torch.arctanh(torch.clamp((x_init - x_orig) / eps, -0.9999, 0.9999))
    w = w_init.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([w], lr=alpha)
    person_class_idx = 0  # Index de la classe 'person'

    for iteration in range(num_iter):
        # Reprojection via tanh pour garantir le respect strict du budget L-inf
        delta = eps * torch.tanh(w)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        
        total_loss = 0.0
        distortion_term = torch.mean(delta ** 2)

        # Calcul cumulé de la perte d'évasion C&W sur l'ensemble des modèles YOLO
        for model in models_list:
            model.model.to(device)
            model.model.eval()
            
            outputs = model.model(x_adv)
            try:
                pred = outputs[0]
                raw_scores = pred[:, 4 + person_class_idx, :]
                person_probs = torch.sigmoid(raw_scores)
                max_person_score = torch.max(person_probs)
            except Exception:
                max_person_score = outputs[0].mean()
                
            total_loss += max_person_score

        # Fonction de perte globale combinant la distorsion L2 et l'objectif d'évasion pondéré par c
        loss = distortion_term + c * total_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(
                f'Itération {iteration+1}/{num_iter} '
                f'| Loss globale={loss.item():.4f} '
                f'| Distorsion={distortion_term.item():.6f} '
                f'| Score Max Personne (Moyenne Ensemble)={total_loss.item() / len(models_list):.6f}'
            )

        torch.cuda.empty_cache()

    with torch.no_grad():
        delta = eps * torch.tanh(w)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
    return x_adv.detach()

def build_adv_filename(original_path):
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Génère un nom de fichier standardisé au format .png pour l'image générée.
    
    ENTRÉES :
        - original_path (str) : Chemin d'accès du fichier image original.
        
    SORTIE :
        - str : Le nom de fichier formaté incluant le préfixe de la méthode et l'extension .png.
    ===========================================================================
    """
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    return f'ensemble_yolo_cw_linf_{name}.png'

def main():
    """
    ===========================================================================
    RÔLE PRINCIPAL :
    Coordonne le pipeline global d'exécution : chargement des 10 modèles YOLO, lecture 
    robuste des images cibles (avec support anti-AVIF via Pillow), affichage des prédictions 
    initiales, lancement de l'attaque C&W conjointe, sauvegarde sans perte au format PNG, 
    et évaluation finale exhaustive sur l'ensemble des modèles.
    
    ENTRÉES :
        - Aucune (utilise les configurations et chemins définis en interne).
        
    SORTIE :
        - Aucune (écrit les images générées sur le disque et affiche les visualisations graphiques).
    ===========================================================================
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device :', device)

    # Ensemble exclusif des modèles YOLO (sans Transformers)
    ensemble_model_names = [
        'yolov5n.pt', 'yolov5x.pt',
        'yolov8n.pt', 'yolov8x.pt',
        'yolo11n.pt', 'yolo11x.pt',
        'yolo12n.pt', 'yolo12x.pt',
        'yolo26n.pt', 'yolo26x.pt'
    ]

    print("Chargement de l'ensemble de modèles YOLO en mémoire...")
    models_list = []
    for name in ensemble_model_names:
        try:
            m = YOLO(name)
            m.model.to(device)
            models_list.append(m)
            print(f"-> Chargé avec succès : {name}")
        except Exception as e:
            print(f"Attention, impossible de charger {name} ({e})")

    if not models_list:
        print("Erreur critique : Aucun modèle n'a pu être chargé.")
        return

    input_paths = [
        '/content/4.avif'
    ]

    images = []
    valid_paths = []

    for path in input_paths:
        if not os.path.exists(path):
            print(f"Fichier non trouvé : {path}")
            continue
        try:
            # Chargement robuste gérant les fichiers .avif récalcitrants via Pillow
            if path.lower().endswith('.avif'):
                pil_img = Image.open(path).convert('RGB')
                img = np.array(pil_img)
            else:
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    continue
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            img_resized = cv2.resize(img, (640, 640))
            images.append(img_resized)
            valid_paths.append(path)
        except Exception as e:
            print(f"Erreur de chargement pour {path}: {e}")

    if not images:
        print("Aucune image valide à traiter.")
        return

    output_dir = '/content/adversarial_images'
    os.makedirs(output_dir, exist_ok=True)

    print('\n===== PREDICTIONS INITIALES (Test sur YOLOv26x de référence) =====')
    ref_model = YOLO('yolo26x.pt')
    ref_model.model.to(device)

    for idx, image in enumerate(images):
        print(f'\nImage {os.path.basename(valid_paths[idx])}')
        results = ref_model.predict(source=image, conf=0.25, verbose=False)[0]
        pred_cls, pred_boxes, _ = extract_predictions(results)
        plot_image_with_boxes(image, pred_boxes, pred_cls)

    # Préparation du tenseur au format de batch [N, C, H, W] et normalisation [0, 1]
    image_stacked = np.stack(images, axis=0).astype(np.float32)
    image_chw = np.transpose(image_stacked, (0, 3, 1, 2))
    image_tensor = torch.tensor(image_chw / 255.0, dtype=torch.float32, device=device)

    print("\n===== GENERATION DE L'ATTAQUE C&W ENSEMBLE (YOLO Only + Linf) =====")
    image_adv_tensor = ensemble_cw_linf_attack(
        models_list=models_list,
        image_tensor=image_tensor,
        eps=32/255,
        alpha=2e-2,
        num_iter=200,
        c=100
    )

    image_adv_chw = image_adv_tensor.detach().cpu().numpy()
    image_adv = np.transpose(image_adv_chw, (0, 2, 3, 1))
    image_adv = np.clip(image_adv * 255, 0, 255).astype(np.uint8)

    for i in range(image_adv.shape[0]):
        img_rgb = image_adv[i]
        file_name = build_adv_filename(valid_paths[i])
        save_path = os.path.join(output_dir, file_name)
        
        # Sauvegarde stricte sans perte au format PNG (conversion RGB vers BGR pour OpenCV)
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f'Image adversariale sauvegardée sans perte : {save_path}')

        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f'Adversarial C&W YOLO Ensemble - {os.path.basename(valid_paths[i])}')
        plt.axis('off')
        plt.show()

    print('\n===== EVALUATION FINALE EXHAUSTIVE APRES ATTAQUE C&W ENSEMBLE =====')
    for idx, image_adv_i in enumerate(image_adv):
        print(f'\n--- Évaluation pour l\'image : {os.path.basename(valid_paths[idx])} ---')
        for m_name in ensemble_model_names:
            try:
                eval_model = YOLO(m_name)
                eval_model.model.to(device)
                results_adv = eval_model.predict(source=image_adv_i, conf=0.25, verbose=False)[0]
                num_persons = sum(1 for box in results_adv.boxes if results_adv.names[int(box.cls.cpu().numpy()[0])] == "person")
                print(f"Modèle {m_name:<12} -> Personnes détectées : {num_persons}")
                
                del eval_model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Erreur avec {m_name}: {e}")

    # Nettoyage final des modèles de la mémoire VRAM
    for m in models_list:
        del m
    del ref_model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()