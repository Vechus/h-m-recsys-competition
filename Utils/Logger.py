import telegram
import os
from dotenv import load_dotenv

class Logger():

    def __init__(self, logger_name='Logger'):
        self.name = logger_name
        load_dotenv()
        self.token = os.getenv('TELEGRAM_API_KEY')
        self.chatid = os.getenv('TELEGRAM_GROUP_CHAT_ID')
        self.bot = telegram.Bot(token=self.token)

    def log(self, message):
        self.bot.send_message(text='[{}] {}'.format(self.name, str(message)), chat_id=self.chatid)
        print('[{}] {}'.format(self.name, str(message)))
