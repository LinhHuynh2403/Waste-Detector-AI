# ♻️ Trash Sorting AI (Recyclable vs Non-Recyclable)
UC Davis Computer Vision project that uses a pretrained CNN to classify waste (plastic, glass, paper, etc.) into recyclable vs. non-recyclable for promoting environmental sustainability.

## 🧠 Project Overview
This project aims to demonstrate how **AI can support environmental sustainability** by automatically classifying waste items (plastic, glass, paper, cardboard, metal, etc.).  
Instead of building a model from scratch, it uses **transfer learning** with pretrained models like **ResNet18** or **MobileNetV2** for efficient and accurate classification.

---

## 📊 Dataset
**Dataset:** [TrashNet](https://www.kaggle.com/code/lutfianaorinnijhum/trashnet-cnn/notebook)  
**Dataset:** [TrashNet V2](https://www.kaggle.com/code/akhileshs111111/trashnet-v2/input)  
- Contains 6 waste categories: *plastic, paper, metal, glass, cardboard, trash*  
- Used to train a model that outputs either:
  - 6-class waste type classification, or  
  - Simplified **recyclable vs non-recyclable** labels  

*(Note: Dataset not included in the repo due to size limits. Please download it from Kaggle.)*

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Resize and normalize images  
   - Apply data augmentation (rotation, flip, brightness)  

2. **Model Training**
   - Use a pretrained CNN (ResNet18 / MobileNetV2)  
   - Fine-tune the final layers for our dataset  
   - Train using categorical cross-entropy loss  

3. **Evaluation**
   - Accuracy and confusion matrix  
   - Grad-CAM visualization for interpretability  

4. **Output**
   - Classifies image as *Recyclable* or *Non-Recyclable*  

---

## 💻 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/LinhHuynh2403/Waste-Detector-AI.git
cd trash-sorting-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training or test script
python train.py     # for training
python predict.py   # for testing new images
