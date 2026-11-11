import json
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def detect_and_export_to_json(image_path, model_path, output_json_path, threshold=0.5):
    """
    DETR modellel detektál egy képen, és az eredményeket JSON fájlba menti.

    Args:
        image_path: A bemeneti kép elérési útja.
        model_path: A betanított DETR modell elérési útja vagy HF azonosítója.
        output_json_path: A kimeneti JSON fájl útvonala.
        threshold: Megbízhatósági küszöbérték a detektálások szűréséhez.
    """
    # 1. Kép és processzor betöltése
    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForObjectDetection.from_pretrained(model_path).eval()

    # 2. Kép előfeldolgozása és detektálás futtatása
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Eredmények utófeldolgozása (normalizált koordinátákból képpontokká)
    # A post_process_object_detection visszaadja a küszöb feletti találatokat
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([image.size[::-1]])  # (magasság, szélesség)
    )[0]

    # 4. JSON struktúra felépítése
    detections_list = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        # A box koordináták: [x_min, y_min, x_max, y_max] képpontban
        box_coords = [round(coord, 2) for coord in box.tolist()]

        # Az osztály neve a modell konfigurációjából (id2label)
        class_name = model.config.id2label[label_id.item()]

        detection = {
            "class_name": class_name,
            "confidence": round(score.item(), 4),
            "bbox": {
                "x_min": box_coords[0],
                "y_min": box_coords[1],
                "x_max": box_coords[2],
                "y_max": box_coords[3]
            }
        }
        detections_list.append(detection)

    final_output = {
        "image_file": image_path,
        "detected_objects_count": len(detections_list),
        "detections": detections_list
    }

    # 5. Mentés JSON fájlba
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Detektálási eredmények mentve: {output_json_path}")
    return final_output


# --- Használati példa ---
if __name__ == "__main__":
    # Cseréld ki a saját finomhangolt modelled elérési útjára!
    # Például: "sajat_modellek/detr-vazlat-sakk" vagy "felhasznalonev/detr-boardgame-sketches"
    MODEL_PATH = "facebook/detr-resnet-50"  # ← Ez csak az általános COCO modell!

    eredmeny = detect_and_export_to_json(
        image_path="kezzel_rajzolt_sakk.jpg",
        model_path=MODEL_PATH,
        output_json_path="detektalas_eredmeny.json",
        threshold=0.5
    )