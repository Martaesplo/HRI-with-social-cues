# HRI-with-social-cues

We investigate the effect of incorporating social cues into Human-Robot Interaction (HRI), aiming towards a more engaging, natural conversation flow.

### Main files:
- **scratch.py:** Contains the ASR, emotion recognition and conversational code. File to be ran on Misty. Change IP address by near the end of the file.
- **emotion_recognition/HRI_emotion_recognition.ipynb:** Code to run and train the emotion recognition model.
- **checkpoint:** Bleongs to the checkpoint(4) used for emotion recognition. Without 'same' padding or data augmentation, and using both datasets.

### Datasets:
- **CREMA-D:** https://github.com/CheyneyComputerScience/CREMA-D.
- **RAVDESS:** https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio.
