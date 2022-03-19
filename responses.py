import re


def process_message(message, response_array, response):
    list_message = re.findall(r"[\w']+|[.,!?;]", message.lower())

    # Score
    score = 0
    for word in list_message:
        if word in response_array:
            score = score + 1

    # returns the response and the score of the response
    return [score, response]


def get_response(message):
    # Add custom response here
    response_list = [
        process_message(message, ['hello', 'hi', 'hey'], 'Hey there! Welcome to DigiFarm, use /start to know more'),
        process_message(message, ['bye', 'goodbye', 'quit'], 'Good Bye!'),
        process_message(message, ['how', 'are', 'you'], 'I\'m doing fine, thanks!'),
        process_message(message, ['who', 'are', 'you'], 'I am DigiFarm, use /start to know more about me'),
        process_message(message, ['your', 'name'],
                        'I am DigiFarm, use /start to know more about me'),
    ]

    # check and return best response by score
    response_scores = []
    for response in response_list:
        response_scores.append(response[0])

    winning_response = max(response_scores)
    matching_response = response_list[response_scores.index(winning_response)]

    if winning_response == 0:
        bot_response = 'Sorry I am unable to understand.\n\nक्षमा करें मैं समझ नहीं पा रहा हूँ\।\n\nਮਾਫ਼ ਕਰਨਾ ਮੈਂ ਸਮਝਣ ਵਿੱਚ ਅਸਮਰੱਥ ਹਾਂ।'

    else:
        bot_response = matching_response[1]

    return bot_response