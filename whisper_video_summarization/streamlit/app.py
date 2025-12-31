import os
from pathlib import Path

import requests
import streamlit as st

from whisper_video_summarization.utils.paths import get_paths

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def inference_page():
    st.header("Инференс видео")

    uploaded_file = st.file_uploader("Загрузите файл", type=["mp4", "wav", "mp3", "mkv"])

    if uploaded_file:
        tmp_dir = Path("/app/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        video_path = tmp_dir / f"{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(str(video_path))

        if st.button("Запустить транскрибацию и суммаризацию"):
            response = requests.post(
                f"{API_URL}/infer/video", json={"path": str(video_path)}, timeout=600
            )

            if response.status_code == 200:
                result = response.json()
                st.subheader("Транскрипция")
                st.write(result["transcription"])

                st.subheader("Суммаризация")
                st.write(result["summary"])
            else:
                st.error(f"Ошибка: {response.status_code}")
                st.error(response.text)


def training_page():
    st.header("Обучение суммаризатора")

    dataset = st.file_uploader("Загрузите датасет (Gazeta)", type=["jsonl", "csv"])
    config_path = st.text_input("Путь к конфигу hydra", "configs/train.yaml")

    if st.button("Запустить обучение"):
        dataset_path = None

        if dataset is not None:
            paths = get_paths()
            upload_dir = Path(paths.gazeta_data_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)

            dataset_path = upload_dir / dataset.name
            with open(dataset_path, "wb") as f:
                f.write(dataset.getbuffer())

        payload = {"config_path": config_path}
        if dataset_path is not None:
            payload["dataset_path"] = str(dataset_path)

        response = requests.post(
            f"{API_URL}/train",
            json=payload,
        )
        st.success("Обучение запущено")
        st.json(response.json())


def main():
    st.sidebar.title("Режим")

    mode = st.sidebar.radio(
        "Выберите режим",
        ["Инференс видео", "Обучение"],
    )

    if mode == "Инференс видео":
        inference_page()
    else:
        training_page()


if __name__ == "__main__":
    main()
