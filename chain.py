import argparse
import base64
import os
import validators
from dotenv import load_dotenv, find_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def make_image_url(image_path):
    image_url = None
    if validators.url(image_path):
        image_url = image_path
    elif os.path.isfile(image_path):
        base64_image = encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"
    assert image_url, "Image URL was not a valid system path or URL"
    return image_url


def get_image_description(image_path):
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY env must be set"

    system_prompt = """
        You are helping a user block advertising on their TV screen while
        watching the TV show, sporting event or movie they wish to watch.
        """
    system_message = SystemMessage(content=system_prompt)

    user_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Is this image from the content I am interested in watching, or an advertsement?",
            },
            {"type": "image_url", "image_url": {"url": make_image_url(image_path)}},
        ],
    )
    messages = [system_message, user_message]
    llm = ChatOpenAI(
        openai_api_key=api_key, model="gpt-4-vision-preview", max_tokens=1500
    )
    llm.invoke(messages)
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Load an image file.")

    parser.add_argument("image", type=str, help="Path to the image file")
    args = parser.parse_args()
    image = args.image
    get_image_description(image)
