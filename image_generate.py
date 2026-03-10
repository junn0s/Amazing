from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def generate_image(prompt: str, n: int = 1, size: str = "1024x1024") -> str:
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=n,
            size=size,
        )
        return response.data[0].url
    except Exception as exc:
        print("이미지 생성 중 오류 발생:", exc)
        return ""
