from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


if __name__ == "__main__":
    audio_files = ["test.wav"]
    lang = ["cmn_Hans"]
    model_card = "omniASR_CTC_300M_v2"  # omniASR_CTC_300M_v2, omniASR_LLM_300M_v2
    batch_size = 1

    # 建立 pipeline + 推論
    pipeline = ASRInferencePipeline(model_card=model_card)
    transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=batch_size)
    for text in transcriptions:
        print(f"{text}")
