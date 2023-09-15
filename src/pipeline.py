
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO

class PredPipe():
    def __init__(self, model, image, noise, blur):
        self.model = model
        self.image = image
        self.path = image.split('/')[-1].split('.')[0]
        self.results = {}
        self.model_scores = []
        self.noise_and_blur = {
            'noise_pctg': noise,
            'blur_pctg': blur
        }
    
    def add_noise_and_blur(self):
        img = cv2.imread(self.image)
        noise = np.random.normal(0, self.noise_and_blur['noise_pctg'], img.shape)git reset --hard HEAD~1
        noise = noise.astype('uint8')
        img = cv2.add(img, noise)
        img = cv2.blur(img, (self.noise_and_blur['blur_pctg'], self.noise_and_blur['blur_pctg']))
        cv2.imwrite(f"../data/processed/inputs/{self.path}_noised.jpg", img)

        return f"../data/processed/noisy/{self.path}_noised.jpg"
    
    def predict_image(self, add_noise=False):
        if add_noise:
            self.image = self.add_noise()

        results = self.model.predict(self.image)
        result = results[0]
        #save model scores
        for box in result.boxes:
            self.model_scores.append(box.conf[0].item())
        img = cv2.imread(self.image)
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            cv2.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{class_id} {conf}", (cords[0], cords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #write predicted image to data/processed/outputs
            cv2.imwrite(f"../data/processed/outputs/{self.path}_predicted.jpg", img)
        return img, self.model_scores
    
    def plot_scores(self):
        plt.plot(self.model_scores)
        plt.title('Model Scores for each noise level')
        plt.xlabel('Noise level')
        plt.ylabel('model score')
        plt.savefig(f"../data/processed/scores/{self.path}_scores.png")
        plt.close()
        return None

    def score(self):
        return self.model_scores
    







        