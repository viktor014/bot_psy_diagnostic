import asyncio
import logging
import sys
import sqlite3
from os import getenv
from config_reader import config
from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from googletrans import Translator
from text_analyze import Predictor1
from predict_audio import predict_audio, preprocess_audio, convert_ogg_to_wav, recognize_speech
from gpt import response_gpt

# Bot token can be obtained via https://t.me/BotFather
TOKEN = getenv("BOT_TOKEN")

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()

# Подключение к базе данных SQLite
conn = sqlite3.connect('depression_data.db')
cursor = conn.cursor()

# Создание таблиц
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        registration_date DATE DEFAULT CURRENT_DATE
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS TextMessages (
        message_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        message_text TEXT,
        depression_percentage FLOAT,
        audio_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES Users(user_id),
        FOREIGN KEY (audio_id) REFERENCES AudioMessages(audio_id)
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS AudioMessages (
        audio_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        audio_path TEXT,
        depression_percentage FLOAT,
        FOREIGN KEY (user_id) REFERENCES Users(user_id)
    )
''')

conn.commit()
conn.close()


# Функция для проверки существования пользователя в таблице Users
def user_exists(user_id):
    try:
        conn = sqlite3.connect('depression_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except sqlite3.Error as e:
        print("Ошибка при проверке существования пользователя:", e)


# Функция для добавления пользователя в таблицу Users
def add_user(user_id, username=None):
    try:
        # Проверяем, существует ли пользователь с указанным user_id
        if not user_exists(user_id):
            conn = sqlite3.connect('depression_data.db')
            cursor = conn.cursor()
            # Если пользователь не существует, добавляем его в базу данных
            cursor.execute('INSERT INTO Users (user_id, username) VALUES (?, ?)', (user_id, username))
            conn.commit()
            conn.close()  # Закрываем соединение после выполнения запроса

    except sqlite3.Error as e:
        print("Ошибка при добавлении пользователя:", e)


# Функция для добавления информации о текстовом сообщении в таблицу TextMessages
def add_text_message(user_id, message_text, depression_percentage, audio_id=None):
    try:
        conn = sqlite3.connect('depression_data.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO TextMessages (user_id, message_text, depression_percentage, audio_id) VALUES (?, ?, ?, ?)',
                       (user_id, message_text, depression_percentage, audio_id))
        text_message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return text_message_id
    except sqlite3.Error as e:
        print("Ошибка при добавлении сообщения:", e)


# Функция для добавления информации об аудиосообщении в таблицу AudioMessages
def add_audio_message(user_id, audio_path, depression_percentage):
    try:
        conn = sqlite3.connect('depression_data.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO AudioMessages (user_id, audio_path, depression_percentage) VALUES (?, ?, ?)',
                       (user_id, audio_path, depression_percentage))
        audio_id = cursor.lastrowid  # Получаем id только что добавленной записи
        conn.commit()
        conn.close()
        return audio_id
    except sqlite3.Error as e:
        print("Ошибка при добавлении аудиосообщения:", e)


def get_average_depression(user_id):
    try:
        conn = sqlite3.connect('depression_data.db')
        cursor = conn.cursor()

        # Получаем среднее значение депрессии из таблицы TextMessages
        cursor.execute('SELECT AVG(depression_percentage) FROM TextMessages WHERE user_id = ?', (user_id,))
        text_avg_depression = cursor.fetchone()[0]

        # Получаем среднее значение депрессии из таблицы AudioMessages
        cursor.execute('SELECT AVG(depression_percentage) FROM AudioMessages WHERE user_id = ?', (user_id,))
        audio_avg_depression = cursor.fetchone()[0]
        conn.close()
        # Если нет данных, возвращаем None
        if text_avg_depression is None and audio_avg_depression is None:
            return None

        # Если нет данных в одной из таблиц, возвращаем среднее из другой
        if text_avg_depression is None:
            return audio_avg_depression
        elif audio_avg_depression is None:
            return text_avg_depression

        # Возвращаем среднее значение депрессии из обеих таблиц
        return (text_avg_depression + audio_avg_depression) / 2

    except sqlite3.Error as e:
        print("Ошибка при получении среднего значения депрессии:", e)
        return None


def text_language(text):
    translator = Translator()
    # Проверка исходного языка текста сообщения, если не английсккий,то вызывается функция перевода текста на английский
    if translator.detect(text).lang != 'en':
        text = translator.translate(text, dest='en')
        return text.text
    return text


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f""
                         f"Привет, {html.bold(message.from_user.full_name)}!\n\n"
                         f"Я - бот, специализирующийся на психиатрической диагностике и поддержке. "
                         f"Я здесь, чтобы помочь Вам с любыми вопросами или проблемами, связанными с психическим здоровьем.\n\n"
                         f"Вы можете общаться со мной, делиться своими мыслями и чувствами, и я постараюсь предложить Вам профессиональные советы и стратегии для того, чтобы справиться с проблемами. "
                         f"Помните, что я не заменяю визит к психиатру или другому специалисту, но я всегда здесь, чтобы поддержать и помочь найти путь к лучшему психическому благополучию.\n\n"
                         f"Начнем наше общение! Как я могу помочь сегодня?")


@dp.message(Command("result"))
async def result(message: Message) -> None:
    answer = get_average_depression(message.from_user.id)
    await message.answer(str(answer))

# Хэндлер при получении голосового сообщения
@dp.message(F.voice)
async def voice_msg(message: Message, bot: Bot) -> None:
    await bot.send_chat_action(message.chat.id, "typing")
    # Задержка для имитации времени на "печатание"
    await asyncio.sleep(2)
    # Получаем информацию о голосовом сообщении
    file = await bot.get_file(message.voice.file_id)
    # Указываем путь до голосового сообщения по уникальному id
    save_path_name = f"Voice/{file.file_unique_id}.ogg"
    # Указываем путь до предобработанного голосового сообщения по уникальному id
    pr_audio_path = f"Voice/{file.file_unique_id}_prep.ogg"
    # Указываем путь до преобразованого файла в формат .wav
    pr_audio_path_wav = f"Voice/{file.file_unique_id}_prep.wav"
    # Загружаем файл на компьютер
    await bot.download_file(file.file_path, save_path_name)
    # Вызываем функцию предварительной обработки голосового сообщения
    preprocess_audio(save_path_name, pr_audio_path)
    # Вызываем модель диагностики депрессии в голосе
    answer_audio = predict_audio(pr_audio_path)
    audio_id = add_audio_message(message.from_user.id, pr_audio_path, answer_audio)
    # Вызываем функцию обработки голосового "ogg" в "wav" и сразу получаем распознанный текст
    try:
        convert_ogg_to_wav(pr_audio_path, pr_audio_path_wav)
        default_text = recognize_speech(pr_audio_path_wav)
        response = response_gpt(default_text)
        text = text_language(default_text)
        depression_percentage_text = Predictor1([text])
        add_text_message(message.from_user.id, default_text, depression_percentage_text, audio_id)
        await message.answer(response)
    except Exception as e:
        await message.answer(
            'Мне не удалось распознать текст из голосового сообщения, но я смог его проанализировать на наличие депрессии: ',
            answer_audio)
        print("Произошло исключение:", e)


@dp.message()
async def echo_handler(message: Message, bot: Bot) -> None:
    try:
        response = response_gpt(message.text)
        await bot.send_chat_action(message.chat.id, "typing")
        # Задержка для имитации времени на "печатание"
        await asyncio.sleep(2)
        await message.answer(response)
        text = text_language(message.text)
        depression_percentage_text = Predictor1([text])
        # Получаем id пользователя и его сообщение
        user_id = message.from_user.id
        username = message.from_user.full_name
        # Вызов функции для добавления пользователя в бд и текста сообщения в бд
        add_user(user_id, username)
        add_text_message(user_id, message.text, depression_percentage_text)

    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=config.bot_token.get_secret_value(), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
