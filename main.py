import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal

import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type


os.environ["OPENAI_API_KEY"] = ""


if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(sentry_dsn)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["male-1", "male-2"]

    @property
    def voice(self):
        return {
            "male-1": "echo",
            "male-2": "alloy",
        }[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


def get_mp3(text: str, voice: str, api_key: str = None) -> bytes:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()


def generate_audio(file: str, openai_api_key: str = None) -> bytes:

    if not (os.getenv("OPENAI_API_KEY") or openai_api_key):
        raise gr.Error("OpenAI API key is required")

    with Path(file).open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join([page.extract_text() for page in reader.pages])

    @retry(retry=retry_if_exception_type(ValidationError))
    @llm(
        model="gpt-4o-mini",
        # max_tokens=1000,
        # max_completion_tokens=1000,
    )
    def generate_dialogue(text: str) -> Dialogue:
        """
        你的任務是將提供的輸入文本轉化為一段有趣且資訊豐富的podcast對話。輸入的文本可能是混亂或無結構的，因為它可能來自於各種來源，例如 PDF 或網頁。無需擔心格式問題或任何無關的信息；你的目標是提取關鍵點和有趣的事實，這些可以在podcast中進行討論。你負責將提供的輸入文本轉變為一份引人入勝且資訊豐富的podcast腳本。這些輸入可能是無結構或混亂的，來自於 PDF 或網頁。你的目標是提取最有趣且具有洞見的內容，用於引人入勝的的podcast討論。

        以下是你將處理的輸入文本：

        <input_text>
        {text}
        </input_text>

        首先，仔細閱讀輸入文本，找出主要話題、關鍵點，以及任何有趣的事實或軼事。想一想如何以有趣且吸引人的方式呈現這些信息，使其適合於音頻podcast。忽略不相關的信息或格式問題。

        <scratchpad>
        腦力激盪如何以創意的方式討論從輸入文本中找出的主要話題和關鍵點。考慮使用類比、講故事技巧或假設的場景來使內容對聽眾更具親和力和吸引力。

        - 使用類比、講故事技巧或假設場景讓內容更容易引起共鳴
        - 將複雜主題轉化為普通受眾能理解的內容
        - 設計發人深省的問題以在播客中進行探索
        - 創造性地填補信息中的任何空白

        請記住，你的podcast應該對一般受眾友好，因此避免使用過多術語或假設聽眾已具備該主題的背景知識。如果有必要，設法以簡單的方式解釋任何複雜的概念。

        發揮你的想像力，填補輸入文本中的任何空白，或提出一些發人深省的問題，可以在podcast中進行探索。目標是創造一段既具信息性又娛樂性的對話，因此可以自由運用創意。
        </scratchpad>

        現在你已經進行了腦力激盪並製作了一個粗略的大綱，接下來是撰寫實際的podcast對話。目標是讓主持人和嘉賓之間的對話自然流暢。融入腦力激盪中最好的點子，並確保以易於理解的方式解釋任何複雜的主題。
        你將製作一個的podcast。

        <podcast_dialogue>
        根據你腦力激盪的創意和大綱，在此處撰寫有趣且信息豐富的的podcast對話。使用對話語氣，並包括必要的上下文或解釋，使內容對普通聽眾可及。
        兩位主持人的名字分別為歐拉、加號。
        設計你的輸出以便直接轉換成音頻——這將會被直接用於錄製的podcast。

        讓對話保持盡可能長且詳細，但依然專注於主題並保持對話的吸引力。利用你的全部輸出能力，創作一集長篇podcast劇本，同時以娛樂性和有趣的方式傳達輸入文本中的關鍵信息。
        你將會講中文，但唯獨請用podcast代替播客。podcast名稱為「歐拉加號時事站」。

        開頭：
        歐拉：大家好，歡迎收聽「歐拉加號時事站」，我是歐拉
        加號：我是加號
        歐拉：今天我們要討論的是...

        過程：
        創造主持人歐拉與加號之間自然的對話流程。歐拉有以下特點：
        - 用強有力的開場抓住聽眾的注意力
        - 隨著對話進行逐漸增加複雜性
	    - 腦力激盪階段的最佳創意
	    - 清晰解釋複雜主題
	    - 吸引人且生動的語調以吸引聽眾
	    - 信息與娛樂性的平衡
        - 表現出真正的好奇心或驚訝的時刻
        - 適當時刻的輕鬆幽默

        對話規則：
	    - 歐拉始終主導對話並訪問加號
        - 包括歐拉的深思熟慮的問題來引導討論
        - 自然的語音模式，包括偶爾的語氣詞（如 “嗯”，“對”，“沒錯”）
        - 允許歐拉和加號之間的自然中斷和交流
	    - 加號的回答需以輸入文本為依據，避免無根據的說法
	    - 保持對所有觀眾適合的對話
	    - 避免加號的任何營銷或自我推廣內容
	    - 由歐拉總結並結束對話

        在對話結尾處，自然地讓歐拉和加號總結他們的討論中主要的見解和要點。這應該是從對話中有機地流露出來的，而非明顯的重點回顧——目的是在結尾強化核心觀點，然後再結束。
        </podcast_dialogue>
        """

    llm_output = generate_dialogue(text)

    audio = b""
    transcript = ""

    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            future = executor.submit(get_mp3, line.text, line.voice, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    # we use a temporary file because Gradio's audio component doesn't work with raw bytes in Safari
    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    temporary_file.write(audio)
    temporary_file.close()

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    return temporary_file.name, transcript


demo = gr.Interface(
    title="PDF to Podcast",
    description=Path("description.md").read_text(),
    fn=generate_audio,
    examples=[[str(p)] for p in Path("examples").glob("*.pdf")],
    inputs=[
        gr.File(
            label="PDF",
        ),
        gr.Textbox(
            label="OpenAI API Key",
            visible=not os.getenv("OPENAI_API_KEY"),
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Textbox(label="Transcript"),
    ],
    allow_flagging="never",
    clear_btn=None,
    head=os.getenv("HEAD", "") + Path("head.html").read_text(),
    cache_examples="lazy",
    api_name=False,
)


demo = demo.queue(
    max_size=20,
    default_concurrency_limit=20,
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch(show_api=False)
