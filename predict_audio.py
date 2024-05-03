import librosa
import numpy as np
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import os
from python_speech_features import mfcc
import shutil
import subprocess
import speech_recognition as sr


# Функция для распознавания речи с помощью SpeechRecognition
def recognize_speech(filename):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(filename)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data, language="ru-RU")
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)


# Фукция для преобразования формата .ogg в .wav
def convert_ogg_to_wav(ogg_file, wav_file):
    # Загрузка аудиофайла формата OGG
    audio = AudioSegment.from_ogg(ogg_file)
    # Экспорт аудиофайла в формат WAV
    audio.export(wav_file, format="wav")


# Фукция для предобработки аудиофайла для анализа
def preprocess_audio(input_file, output_file):
    # Полный путь к ffmpeg
    ffmpeg_path = r'D:\Program Files\fmpeg\bin\ffmpeg.exe'  # Укажите полный путь к ffmpeg.exe на вашем компьютере

    # Команда ffmpeg для предварительной обработки аудиофайла (в данном случае - просто копирование аудиодорожки без метаданных)
    command = [ffmpeg_path, '-i', input_file, '-map_metadata', '-1', '-c:a', 'copy', output_file]

    # Запуск команды ffmpeg с помощью subprocess
    try:
        subprocess.run(command, check=True)
        print("Предварительная обработка завершена успешно.")
    except subprocess.CalledProcessError as e:
        print("Ошибка при выполнении предварительной обработки:", e)


def predict_audio(input):
    #папка для сохранения клипов и mfcc
    output_folder = "Bucket"

    # создать папку если не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #загрузить аудио файл
    audio = AudioSegment.from_ogg(input)

    #получить длину аудио
    duration = len(audio)

    #настройка длины аудио
    clip_length = 3000

    #разделение на 5-секундные клипы
    for i in range(0, duration, clip_length):
        start_time = i
        end_time = i + clip_length

        clip = audio[start_time:end_time]
        #сохранить эти клипы в папке мусор
        output_file = os.path.join(output_folder, f"clip{i // clip_length}.ogg")
        clip.export(output_file, format="ogg")

    for file_name in os.listdir(output_folder):
        if file_name.endswith('.ogg'):
            #загрузить аудио файл
            audio_path = os.path.join(output_folder, file_name)
            audio, sr = librosa.load(audio_path, sr=44100)

            #обработка 5 секундных аудио клипов
            duration = 5  # сек
            samples = sr * duration
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            if len(audio) > samples:
                audio = audio[:samples]
            else:
                audio = np.pad(audio, (0, samples - len(audio)), mode='constant')

            #извлечение признаков с помощью mfcc
            mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512)

            #сохранение mfcc
            mfcc_path = os.path.join(output_folder, file_name.replace('.ogg', '.npy'))
            np.save(mfcc_path, mfccs)

    mfccs = []
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.npy'):
            #загрузка mfccs
            mfcc_path = os.path.join(output_folder, file_name)
            mfcc = np.load(mfcc_path)
            # добавление mfccs в список
            mfccs.append(mfcc)

    # преобразовать список в массив numpy
    mfccs = np.array(mfccs)

    #загрузить обученную модель
    model = load_model('Models/audio/model3.h5')

    #делаем прогнозы
    predictions = model.predict(mfccs)

    def average_prediction(predictions):
        total = 0
        for prediction in predictions:
            total += prediction
        return total / len(predictions)

    final_prediction = (average_prediction(predictions)) * 100

    # удалить аудио
    # os.remove(input)
    shutil.rmtree(output_folder)

    #  вывода
    prediction_print = "{:.2f}".format(final_prediction[0])
    return prediction_print
