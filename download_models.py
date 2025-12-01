import requests
import gdown
import os

models_urls = [
    'https://raw.githubusercontent.com/Serurays/Sample_Detection_Tracking_Yolov8_DeepSORT_Robotics/main/mars-small128.pb'
]

google_urls = {
    'osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth': "https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
    'resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth': 'https://drive.google.com/uc?id=1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf',
    'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth': 'https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x',
    'osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth': 'https://drive.google.com/uc?id=14bNFGm0FhwHEkEpYKqKiDWjLNhXywFAd',
    'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth': 'https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA',
    'market1501.pb': 'https://drive.google.com/uc?id=1cTFbJALraAZ6b92r8LjfbyXEvW5iXcta'
}

def download_yolo_model():
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    output = "yolov8n.pt"

    print("Téléchargement en cours...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Fichier téléchargé :", output)


if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.chdir("models")
    for model in models_urls:
        status = os.system(f"curl -O {model}")
        if status != 0:
            print("Erreur : Erreur dans le téléchargement de models.")
            exit(status)
    for model_name, url in google_urls.items():
        gdown.download(url, model_name, quiet=False)

    download_yolo_model()
