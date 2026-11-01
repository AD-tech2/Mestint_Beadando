import json

def generate_pygame_ui(json_file_path, output_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    class_name = data.get("ui_name", "GeneratedUI")
    
    # Kód template inicializálása
    generated_code = f"""import pygame

class {class_name}:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 36)
        self.elements = {{}}
"""
    
    # 1. Init blokk felépítése (Rect objektumok definíciója)
    for el in data["elements"]:
        el_id = el["id"]
        x, y, w, h = el["x"], el["y"], el["w"], el["h"]
        generated_code += f"        self.elements['{el_id}'] = pygame.Rect({x}, {y}, {w}, {h})\n"
        
    # 2. Draw metódus felépítése
    generated_code += "\n    def draw(self, surface):\n"
    for el in data["elements"]:
        el_id = el["id"]
        label = el.get("label", "")
        
        if el["type"] == "button":
            generated_code += f"        pygame.draw.rect(surface, (70, 70, 70), self.elements['{el_id}'])\n"
            if label:
                generated_code += f"        text_surf = self.font.render('{label}', True, (255, 255, 255))\n"
                generated_code += f"        surface.blit(text_surf, (self.elements['{el_id}'].x + 10, self.elements['{el_id}'].y + 10))\n"
                
        elif el["type"] == "health_bar":
            generated_code += f"        pygame.draw.rect(surface, (200, 0, 0), self.elements['{el_id}'])\n"
            generated_code += f"        pygame.draw.rect(surface, (255, 255, 255), self.elements['{el_id}'], 2)\n"

    # Fájl kiírása
    with open(output_file_path, 'w') as f:
        f.write(generated_code)
        
    print(f"Siker! A {output_file_path} legenerálva.")

# Futtatás
generate_pygame_ui('layout.json', 'generated_ui.py')