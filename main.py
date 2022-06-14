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

	dis = ['The disease is Leaf rust\n\n1. Rust loves damp conditions, so avoid overwatering your plants.\n2.Look for rust-resistant cultivars of the plants you want to grow.\n\nबीमारी है लीफ रस्ट\n\n1. जंग नम स्थितियों से प्यार करती है, इसलिए अपने पौधों को अधिक पानी देने से बचें।\n2. उन पौधों की जंग प्रतिरोधी किस्मों की तलाश करें जिन्हें आप उगाना चाहते हैं।ਬਿਮਾਰੀ ਪੱਤਿਆਂ ਦੀ ਜੰਗਾਲ ਹੈ\n\n1. ਜੰਗਾਲ ਗਿੱਲੀ ਸਥਿਤੀਆਂ ਨੂੰ ਪਸੰਦ ਕਰਦਾ ਹੈ, ਇਸਲਈ ਆਪਣੇ ਪੌਦਿਆਂ ਨੂੰ ਜ਼ਿਆਦਾ ਪਾਣੀ ਦੇਣ ਤੋਂ ਬਚੋ।\n2.ਜਿਨ੍ਹਾਂ ਪੌਦਿਆਂ ਨੂੰ ਤੁਸੀਂ ਉਗਾਉਣਾ ਚਾਹੁੰਦੇ ਹੋ ਉਨ੍ਹਾਂ ਦੀਆਂ ਜੰਗਾਲ-ਰੋਧਕ ਕਿਸਮਾਂ ਦੀ ਭਾਲ ਕਰੋ।','The disease is Stem rust\n\n1.reducing the inoculum in a district by managing the green bridge\n\n2.Close monitoring to enable timely fungicide sprays.\n\nरोग है स्टेम रस्ट\n\n1. हरित पुल का प्रबंधन करके जिले में इनोकुलम को कम करना\n\n2. समय पर कवकनाशी स्प्रे सक्षम करने के लिए निगरानी बंद करें।\n\nਬਿਮਾਰੀ ਸਟੈਮ ਰਸਟ ਹੈ\n\n1. ਹਰੇ ਪੁਲ ਦਾ ਪ੍ਰਬੰਧਨ ਕਰਕੇ ਜ਼ਿਲ੍ਹੇ ਵਿੱਚ ਇਨੋਕੁਲਮ ਨੂੰ ਘਟਾਉਣਾ\n\n2. ਸਮੇਂ ਸਿਰ ਉੱਲੀਨਾਸ਼ਕ ਸਪਰੇਆਂ ਨੂੰ ਸਮਰੱਥ ਬਣਾਉਣ ਲਈ ਨਿਗਰਾਨੀ ਬੰਦ ਕਰੋ।\n\n', 'The disease is Yellow rust\n\n1.SDHIs used for septoria control will also give control of yellow rust.\n\n2.Grow a variety with a high resistance rating, but monitor disease levels throughout the season\n\nरोग है पीला रतुआ\n\n1. सेप्टोरिया नियंत्रण के लिए उपयोग किए जाने वाले SDHI भी पीले रतुआ का नियंत्रण देंगे।\n\n2. उच्च प्रतिरोध रेटिंग वाली किस्म उगाएं, लेकिन पूरे मौसम में रोग के स्तर की निगरानी करें\n\nਬਿਮਾਰੀ ਪੀਲੀ ਕੁੰਗੀ ਹੈ\n\n1. ਸੈਪਟੋਰੀਆ ਨਿਯੰਤਰਣ ਲਈ ਵਰਤੀਆਂ ਜਾਣ ਵਾਲੀਆਂ SDHIs ਪੀਲੀ ਕੁੰਗੀ ਨੂੰ ਵੀ ਨਿਯੰਤਰਿਤ ਕਰਨਗੀਆਂ।\n\n2. ਉੱਚ ਪ੍ਰਤੀਰੋਧ ਰੇਟਿੰਗ ਨਾਲ ਕਈ ਕਿਸਮਾਂ ਨੂੰ ਉਗਾਓ, ਪਰ ਪੂਰੇ ਸੀਜ਼ਨ ਦੌਰਾਨ ਬਿਮਾਰੀ ਦੇ ਪੱਧਰਾਂ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ\n\n','The disease is Powdery mildew\n\n1.Use sulfur-containing organic fungicides as both preventive and treatment for existing infections.\n\n2.Thin out existing susceptible plants to improve airflow within the plant.\n\nरोग ख़स्ता फफूंदी है\n\n1। मौजूदा संक्रमणों के लिए निवारक और उपचार दोनों के रूप में सल्फर युक्त कार्बनिक कवकनाशी का उपयोग करें।\n\n2। संयंत्र के भीतर वायु प्रवाह में सुधार के लिए मौजूदा अतिसंवेदनशील पौधों को पतला करें।\n\nਇਹ ਬਿਮਾਰੀ ਪਾਊਡਰਰੀ ਫ਼ਫ਼ੂੰਦੀ ਹੈ\n\n1.ਮੌਜੂਦਾ ਲਾਗਾਂ ਦੀ ਰੋਕਥਾਮ ਅਤੇ ਇਲਾਜ ਦੇ ਤੌਰ ਤੇ ਗੰਧਕ ਵਾਲੇ ਜੈਵਿਕ ਉੱਲੀਨਾਸ਼ਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰੋ।\n\n2.ਪੌਦੇ ਦੇ ਅੰਦਰ ਹਵਾ ਦੇ ਪ੍ਰਵਾਹ ਨੂੰ ਬਿਹਤਰ ਬਣਾਉਣ ਲਈ ਮੌਜੂਦਾ ਸੰਵੇਦਨਸ਼ੀਲ ਪੌਦਿਆਂ ਨੂੰ ਪਤਲਾ ਕਰੋ।\n\n', 'The disease is septoria\n\n1.Water aids the spread of Septoria leaf spot. Keep it off the leaves as much as possible by watering at the base of the plant only.\n\n2.Improve air circulation around the plants.\n\nरोग है सेप्टोरिया\n\n1. पानी सेप्टोरिया लीफ स्पॉट के प्रसार में सहायता करता है। केवल पौधे के आधार पर पानी देकर इसे जितना हो सके पत्तियों से दूर रखें।\n\n2.पौधों के चारों ओर वायु परिसंचरण में सुधार करें।\n\nਇਹ ਬਿਮਾਰੀ ਸੇਪਟੋਰੀਆ ਹੈ\n\n1. ਪਾਣੀ ਸੇਪਟੋਰੀਆ ਦੇ ਪੱਤਿਆਂ ਦੇ ਧੱਬੇ ਨੂੰ ਫੈਲਾਉਣ ਵਿੱਚ ਸਹਾਇਤਾ ਕਰਦਾ ਹੈ। ਪੌਦੇ ਦੇ ਅਧਾਰ ਤੇ ਪਾਣੀ ਦੇ ਕੇ ਜਿੰਨਾ ਸੰਭਵ ਹੋ ਸਕੇ ਇਸ ਨੂੰ ਪੱਤਿਆਂ ਤੋਂ ਦੂਰ ਰੱਖੋ।\n\n2.ਪੌਦਿਆਂ ਦੇ ਆਲੇ ਦੁਆਲੇ ਹਵਾ ਦੇ ਗੇੜ ਵਿੱਚ ਸੁਧਾਰ ਕਰੋ।\n\n']

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