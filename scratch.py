# Add mistyPy directory to sys path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mistyPy.Robot import Robot
from mistyPy.Events import Events
import speech_recognition as sr
import base64
import librosa
import resampy
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers

checkpoint_path = 'cp.ckpt'

#towards the full automated robot
def first_sentence():
    misty.speak("I'm going to show a picture. Do you recognize this place?", 1, 1, None, True, "tts-content")
    misty.display_image("Barcelona.jpg")

def start_speech_recognition():
    # misty.speak("I'm going to show a picture. Do you recognize this place?", 1.2, 1.2, None, True, "tts-content")
    # misty.display_image("e_DefaulContent.jpg")

    misty.register_event(Events.TextToSpeechComplete, "initTTSComplete", keep_alive=False, callback_function=tts_intro_completed)

    # misty.display_image("e_DefaultContent.jpg")
    misty.move_head(0, 0, 0, 50)
    misty.move_arms(12, -12, 1, 1, 1)

def tts_intro_completed(event):
    # keep_alive defaults to false
    misty.register_event(Events.TextToSpeechComplete, "whatDoYouSeeTTSComplete", callback_function=tts_what_do_you_see_completed)
    misty.move_arms(-12, 12, 1, 1, 1)
    # misty.speak("Do you recognize the place?", 1.2, 1.2, None, True, "tts-content")

def tts_what_do_you_see_completed(event):
    # json = {
    #         "overwriteExisting": True,
    #         "silenceTimeout": 2000,
    #         "maxSpeechLength": 15000,
    #         "requireKeyPhrase": False,}
    
    misty.capture_speech(True, 3000, 15000, False, "en-us")
    misty.register_event(Events.VoiceRecord, "VoiceRecord", callback_function=voice_record_complete)
    # misty.CaptureSpeechAzure(True, 2000, 15000, False, False, "en-us", "<azure_cognitive_services_key>", "eastus")
    # misty.post_request('audio/speech/capturevosk', json=json)
    # misty.display_image("e_DefaultContent.jpg")
    misty.move_head(0, 0, 0, 50)


def voice_record_complete(event):
    print(event)
    if "message" in event:
        print(event)
        parsed_message = event["message"]
        misty_heard = parsed_message["speechRecognitionResult"]
        print(f"Misty heard: {misty_heard}")

     # use the audio file as the audio source
    r = sr.Recognizer()
    audiofile = misty.get_audio_file(parsed_message['filename'], True).json()
    wav_file = open("temp.wav", "wb")
    decode_string = base64.b64decode(audiofile['result']['base64'])
    wav_file.write(decode_string)
    import simpleaudio as sa

    filename = wav_file.name
    wave_obj = sa.WaveObject.from_wave_file(filename)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()
    start_emotion_recognition()

    with sr.AudioFile(filename) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
        text = "";
        try:
            text = r.recognize_google(audio)
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            print("Google Speech Recognition thinks you said " + text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    # do something with this data
    # misty.move_head(-30, 20, -50, 85, None, None)
    misty.move_arms(89, 89, 1, 1, 1)
    misty.register_event(Events.TextToSpeechComplete, "finalTTSComplete", callback_function=tts_all_i_ever_see)
    # misty.speak("You said" + text, 1.2, None, None, True, "tts-content")

def tts_all_i_ever_see(event):
    misty.display_image("e_Joy.jpg")

def start_emotion_recognition():
    
    labels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

    features=True
    #train=False
    load=True
    predict=True
    tree=False

    raw_audio = misty.get_audio_file("capture_Dialogue.wav", True).json()
    wav_file = open("temp.wav", "wb")
    decode_string = base64.b64decode(raw_audio['result']['base64'])
    wav_file.write(decode_string)

    audio_file = wav_file.name

    def mel_spectrogram(audio_file):
        X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
        #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        #temporally average spectrogram
        log_spectrogram = np.mean(db_spec, axis = 0)
        return log_spectrogram

    if features:
        mel_spec_audio=mel_spectrogram(audio_file)
        X_test111 = mel_spec_audio
        n1=259-X_test111.shape[0] #259 being the input shape of the trained model
        n=n1//2
        X_test111=np.pad(X_test111, n, 'constant') #padding the number of mfccs missing
        mean = np.mean(X_test111, axis=0) #normalizing
        std = np.std(X_test111, axis=0)
        X_test111 = (X_test111 - mean)/std
        X_test111=X_test111[np.newaxis,:,np.newaxis] #adding missing dimentions

    #ADD X_TRAIN AND Y_TRAIN
    #if train:
    # with open(audio_file) as audio_file:
    #  checkpoint_path = 'cp.ckpt'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        #model_history=model.fit(X_train, y_train, batch_size=36, epochs=100, validation_data=(X_test, y_test), callbacks=[cp_callback])

    if load:
        def cnn_model(X_train, optimizer):
            model = tf.keras.Sequential()
            model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
            model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(layers.MaxPooling1D(pool_size=(8)))
            model.add(layers.Dropout(0.4))
            model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=(8)))
            model.add(layers.Dropout(0.4))
            model.add(layers.Flatten())
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(6, activation='softmax'))
            if optimizer=='Adam': opt = keras.optimizers.Adam(learning_rate=0.001)
            elif optimizer=='SGD': opt = keras.optimizers.SGD(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
            model.summary()
            return model

    model = cnn_model(X_test111,'SGD')

    #loading model weights from checkpoint
    model.load_weights(checkpoint_path)

    if predict:
        predictions = model.predict(X_test111)
        #second highest:
        predictions2 = np.sort(predictions)
        predictions2 = np.argsort(predictions[0])[::-1][:2]
        pred_label1 = predictions2[0]
        pred_label2 = predictions2[1]
        pred_val1 = predictions[0][predictions2][0]*100
        pred_val2 = predictions[0][predictions2][1]*100
        print(str(labels[pred_label1])+' '+str(round(pred_val1,2))+'%')
        print(str(labels[pred_label2])+' '+str(round(pred_val2,2))+'%')

    if tree:
         from sklearn import tree
         clf = tree.DecisionTreeClassifier()
         clf = clf.fit(X_train, y_train)
         x_test111t=X_test111
         x_test111t=x_test111t[:,:,0]
         predictions1 = clf.predict(x_test111t)
         predictions1 = predictions1.argmax(axis=1)
         predictions1 = predictions1.astype(int).flatten()
         pred_label1 = predictions1[0]
         print(labels[pred_label1])
         print(clf.score(X_test, y_test))

def reset_misty():
    misty.move_head(0, 0, 0)
    misty.move_arms(89, 89)
    misty.display_image("e_DefaultContent.jpg")

def trigger_input():
    return input(str("press Action:"))

def ask_question(qnum: int):
    if qnum == "1":
        misty.speak("Hi, I am Misty and I live in the Netherlands. I love going on holidays. Right now, I am going to show a picture a of a city. Do you recognize this place?", 1.1, 1.1, None, True,
                    "tts-content")
        misty.display_image("Barcelona.jpg")
    elif qnum == "2":
        misty.display_image("e_DefaultContent.jpg")
        misty.speak("This is a picture of Barçelona, one of the biggest cities in Spain. Have you ever been to Spain?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "3":
        misty.speak("It is a beautiful country. Would you like to go there?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "4":
        misty.speak("That must have been wonderful. Did you like it?", 1.1, 1.1,None, True,
                    "tts-content")
    elif qnum == "5":
        misty.speak("Of all the countries in the world, name your favourite country that you have visited?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "6":
        misty.speak("Wow, that sounds lovely. Did you like the food there?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "7":
        misty.speak("Nice. Which was your favourite dish?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "8":
        misty.speak("That is not nice. How so?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "9":
        misty.speak("Is there any other country that you would like to visit for holiday?", 1.1, 1.1, None, True,
                    "tts-content")
    elif qnum == "0":
        misty.speak(" ", 1, 1, None, True,
                    "tts-content")


def react_angry():
    misty.display_image("e_Anger.jpg")
    misty.play_audio("s_Anger.wav")

def react_fear():
    misty.display_image("e_Fear.jpg")
    misty.play_audio("s_Fear.wav")

def react_happy():
    misty.display_image("e_Admiration.jpg")
    misty.play_audio("s_Awe.wav")

def react_disappointed():
    misty.display_image("e_Contempt.jpg")
    misty.play_audio("s_Disgust.wav")

def react_sad():
    misty.display_image("e_Sadness.jpg")
    misty.play_audio('s_Sadness.wav')

if __name__ == "__main__":
    ip_address = "192.168.43.66"
    misty = Robot(ip_address)
    reset_misty()
    while True:
        key = trigger_input()
        if key == ",":
            first_sentence()
        if key == ".":
            start_speech_recognition()
        elif key == "r":
            reset_misty()
        elif key == "x":
            misty.speak("That sounds wonderful. I hope you manage to go there soon. Unfortunately, I have to go now. It has been great to have a conversation with you.", 1.1, 1.1, None, True,
                    "tts-content")
            reset_misty()
            break
        elif key == 'a':
            react_angry()
        elif key == 'f':
            react_fear()
        elif key == 'h':
            react_happy()
        elif key == 'd':
            react_disappointed()
        elif key == 's':
            react_sad()
        else:
            ask_question(key)
            start_speech_recognition()

