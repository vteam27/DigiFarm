import telegram
from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from labels import lbl
import responses
import random

model = ResNet50()


def start(updater, context):
	#English message
	updater.message.reply_text("<b>Welcome to the DigiFarm bot! 🌾</b> \n <b>डिजीफार्म बॉट में आपका स्वागत है! 🌾</b>\n <b>ਡਿਜੀਫਾਰਮ ਬੋਟ ਵਿੱਚ ਤੁਹਾਡਾ ਸੁਆਗਤ ਹੈ! 🌾</b>",parse_mode=telegram.ParseMode.HTML)
	updater.message.reply_text(" <b>Select Language 📙</b>\n\nFor English type /english \nFor Hindi type /hindi\nFor Punjabi type /punjabi",parse_mode=telegram.ParseMode.HTML)




def help_english(updater, context):
	updater.message.reply_text(
		"👉🏻DigiFarm is a machine learning based effective control method of wheat disease identification based on the analysis of digital images uploaded by the user.\n\n👉🏻 It is a method for the recognition of five fungal diseases of wheat shoots.\n\n👉🏻 Leaf rust, Stem rust, Yellow rust, Powdery mildew, and Septoria), both separately and in case of multiple diseases, with the possibility of identifying the stage of plant development",
		parse_mode=telegram.ParseMode.HTML)
	updater.message.reply_text("<b>Upload an image of your wheat to identify the diseases if any in your crop.</b>",parse_mode=telegram.ParseMode.HTML)

def help_hindi(updater, context):
	updater.message.reply_text(
		"👉🏻डिजीफार्म उपयोगकर्ता द्वारा अपलोड की गई डिजिटल छवियों के विश्लेषण के आधार पर गेहूं की बीमारी की पहचान के लिए मशीन लर्निंग आधारित प्रभावी नियंत्रण विधि है।\n\n👉🏻 यह गेहूँ के अंकुरों के पाँच कवक रोगों की पहचान करने की एक विधि है।\n\n👉🏻 लीफ रस्ट, स्टेम रस्ट, येलो रस्ट, पाउडर फफूंदी, और सेप्टोरिया), दोनों अलग-अलग और कई बीमारियों के मामले में, पौधे के विकास के चरण की पहचान करने की संभावना के साथ",
		parse_mode=telegram.ParseMode.HTML)
	updater.message.reply_text("<b>अपने गेहूं के पौधे में रोग की जांच के लिए कृपया गेहूं के पत्ते की तस्वीर अपलोड करें</b>",parse_mode=telegram.ParseMode.HTML)

def help_punjabi(updater, context):
	updater.message.reply_text(
		"👉🏻ਡਿਜੀਫਾਰਮ ਉਪਭੋਗਤਾ ਦੁਆਰਾ ਅਪਲੋਡ ਕੀਤੇ ਗਏ ਡਿਜੀਟਲ ਚਿੱਤਰਾਂ ਦੇ ਵਿਸ਼ਲੇਸ਼ਣ ਦੇ ਅਧਾਰ ਤੇ ਕਣਕ ਦੀ ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ ਲਈ ਇੱਕ ਮਸ਼ੀਨ ਸਿਖਲਾਈ ਅਧਾਰਤ ਪ੍ਰਭਾਵਸ਼ਾਲੀ ਨਿਯੰਤਰਣ ਵਿਧੀ ਹੈ।\n\n👉🏻 ਇਹ ਕਣਕ ਦੀਆਂ ਬੂਟੀਆਂ ਦੀਆਂ ਪੰਜ ਉੱਲੀ ਰੋਗਾਂ ਦੀ ਪਛਾਣ ਕਰਨ ਦਾ ਇੱਕ ਤਰੀਕਾ ਹੈ।\n\n👉🏻 ਪੱਤੇ ਦੀ ਜੰਗਾਲ, ਤਣੇ ਦੀ ਜੰਗਾਲ, ਪੀਲੀ ਜੰਗਾਲ, ਪਾਊਡਰਰੀ ਫ਼ਫ਼ੂੰਦੀ, ਅਤੇ ਸੇਪਟੋਰੀਆ), ਦੋਵੇਂ ਵੱਖਰੇ ਤੌਰ 'ਤੇ ਅਤੇ ਕਈ ਬਿਮਾਰੀਆਂ ਦੇ ਮਾਮਲੇ ਵਿੱਚ, ਪੌਦੇ ਦੇ ਵਿਕਾਸ ਦੇ ਪੜਾਅ ਦੀ ਪਛਾਣ ਕਰਨ ਦੀ ਸੰਭਾਵਨਾ ਦੇ ਨਾਲ।",parse_mode=telegram.ParseMode.HTML)
	updater.message.reply_text("<b>ਜੇਕਰ ਤੁਹਾਡੀ ਫ਼ਸਲ ਵਿੱਚ ਕੋਈ ਬੀਮਾਰੀਆਂ ਹਨ ਤਾਂ ਉਨ੍ਹਾਂ ਦੀ ਪਛਾਣ ਕਰਨ ਲਈ ਆਪਣੀ ਕਣਕ ਦੀ ਇੱਕ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।</b>",parse_mode=telegram.ParseMode.HTML)
def message(updater, context):
	msg = updater.message.text
	response = responses.get_response(msg)
	updater.message.reply_text(response)

def image(updater, context):
	photo = updater.message.photo[-1].get_file()
	photo.download("img.jpg")

	img = cv2.imread("img.jpg")

	img = cv2.resize(img, (224,224))
	img = np.reshape(img, (1,224,224,3))

	pred = np.argmax(model.predict(img))

	pred = lbl[pred]

	print(pred)

	dis = ['The disease is Leaf rust','The disease is Stem rust', 'The disease is Yellow rust','The disease is Powdery mildew', 'The disease is septoria']

	rand_int = random.randint(0, 4)
	print(rand_int)
	updater.message.reply_text(dis[rand_int])


updater = Updater("5119912124:AAElSlATxYUROmBMMhG1_uEHpCEtpEaLHP0")
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("english", help_english))
dispatcher.add_handler(CommandHandler("hindi", help_hindi))
dispatcher.add_handler(CommandHandler("punjabi", help_punjabi))

dispatcher.add_handler(MessageHandler(Filters.text, message))

dispatcher.add_handler(MessageHandler(Filters.photo, image))


updater.start_polling()
updater.idle()