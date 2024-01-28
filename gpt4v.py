import argparse
import base64
import requests
import os
import validators
from dotenv import load_dotenv, find_dotenv


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_description(image_path):
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY env must be set"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    system_message = {
        "role": "system",
        "content": """
            You are helping a user block advertising on their TV screen while
            watching the TV show, sporting event or movie they wish to watch.
            """,
    }

    image_url = None
    if validators.url(image_path):
        image_url = image_path
    elif os.path.isfile(image_path):
        base64_image = encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"
    assert image_url, "Image URL was not a valid system path or URL"

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Is this image from the content I am interested in watching, or an advertsement?",
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            },
        ],
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            system_message,
            user_message
        ],
        "max_tokens": 1000,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    print(response.json())


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Load an image file.")

    parser.add_argument("image", type=str, help="Path to the image file")
    args = parser.parse_args()
    image = args.image
    get_image_description(image)
