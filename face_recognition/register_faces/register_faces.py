import requests
import base64
from PIL import Image

def resize_image(path, min_width=400, min_height=300):
    img = Image.open(path)
    if img.width < min_width or img.height < min_height:
        new_img = img.resize((max(min_width, img.width), max(min_height, img.height)))
        resized_path = f"resized_{path}"
        new_img.save(resized_path)
        return resized_path
    return path

def register_face(user_id, image_path):
    # Resize first
    resized_path = resize_image(image_path)
    
    with open(resized_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    data = {
        "user_id": user_id,
        "image_base64": f"data:image/jpeg;base64,{img_b64}"
    }
    r = requests.post("http://localhost:5000/register", json=data)
    print(f"{user_id}: {r.json()}")

# Use actual PHP-generated user_ids
register_face("USER_68617494a33632.40468956", "Aaron_Sorkin_0002.jpg")
register_face("USER_686174ac3f49b6.15280164", "Aaron_Eckhart_0001.jpg")
register_face("USER_686175d84d5e31.77005444", "Aaron_Guiel_0001.jpg")
register_face("USER_68617652de7d07.81328708", "Aaron_Patterson_0001.jpg")
register_face("USER_68617663527471.07016789", "Aaron_Peirsol_0001.jpg")
