
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import time    

class PredPipe():
    def __init__(self, model, image_path, noise, blur):
        self.model = model
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.name = image_path.split('/')[-1].split('.')[0]
        self.results = {}
        self.model_scores = []
        self.noise_and_blur = {
            'noise_pctg': noise,
            'blur_pctg': blur
        }
        self.timestamp = str(int(time.time())) # add timestamp as unique identifier

    def add_noise_and_blur(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Add noise
        noise = np.random.uniform(low=0.0, high=1.0, size=image.shape)
        noise_mask = np.random.binomial(1, self.noise_and_blur['noise_pctg'], size=image.shape[:-1])
        noise_mask = np.expand_dims(noise_mask, axis=-1)
        noise_mask = np.repeat(noise_mask, 3, axis=-1)
        image = noise_mask * noise + (1 - noise_mask) * image
        image = (image * 255.0).astype(np.uint8)
        noisy_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        noisy_image_path = f"../data/processed/noisy/{self.name}_{str(self.noise_and_blur['noise_pctg'])}_noisy_{self.timestamp}.jpg" # add timestamp to filename
        cv2.imwrite(noisy_image_path, noisy_image)

        return image, noisy_image_path

    def predict_image(self, add_noise=False):
        if add_noise:
            image, noisy_image_path = self.add_noise_and_blur()
            results = self.model.predict(noisy_image_path)
            result = results[0]
            predicted_image_path = f"../data/processed/outputs/{self.name}_{str(self.noise_and_blur['noise_pctg'])}_noisy_predicted_{self.timestamp}.jpg"
        
        else:
            results = self.model.predict(self.image_path)
            result = results[0]
            predicted_image_path = f"../data/processed/outputs/{self.name}_predicted_{self.timestamp}.jpg"
        
        #save model scores
        for box in result.boxes:
            class_name = result.names[box.cls[0].item()]
            class_confidence = round(box.conf[0].item(), 2)
            if class_name not in self.results:
                self.results[class_name] = []
            self.results[class_name].append(class_confidence)
            self.model_scores.append(class_confidence)
                
        img = cv2.imread(self.image_path if not add_noise else noisy_image_path)
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            cv2.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{class_id} {conf}", (cords[0], cords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #write predicted image to data/processed/outputs
            cv2.imwrite(predicted_image_path, img)

        return img, self.results

    def plot_scores(self, save=False):
        # plot model scores
        plt.figure(figsize=(10, 8))
        plt.hist(self.model_scores)
        plt.title(f"{self.name} Model Scores")
        plt.xlabel("Model Scores")
        plt.ylabel("Frequency")
        if save:
            plt.savefig(f"../reports/figures/{self.name}_model_scores.png")
        plt.show()







        