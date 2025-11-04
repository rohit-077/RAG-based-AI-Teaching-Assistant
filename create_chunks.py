import whisper
import json
import os

# Load Whisper model once at the beginning.
# I'm using large-v2 here but you can change it to medium or small if your system is slow.
print("loading whisper model... might take a bit depending on your hardware.")
try:
    model = whisper.load_model("large-v2")
    print("model loaded successfully.")
except Exception as e:
    print("failed to load whisper model:", e)
    exit()

# make sure the converted_texts folder exists
if not os.path.exists("converted_texts"):
    os.makedirs("converted_texts")
    print("created folder: converted_texts")

# loop through all audio files
for audio in os.listdir("audios"):
    # only process files that follow the naming pattern (e.g., 02_Title.mp3)
    if "_" in audio and "01" not in audio:
        filename = os.path.splitext(audio)[0]
        try:
            number, title = filename.split("_", 1)
        except ValueError:
            print(f"skipping {audio} because filename format is not as expected.")
            continue

        audio_path = os.path.join("audios", audio)
        print(f"\nprocessing audio file: {audio_path}")

        try:
            # transcribe the audio
            print("running whisper transcription...")
            result = model.transcribe(audio_path)
        except Exception as e:
            print(f"error while transcribing {audio}: {e}")
            continue

        # prepare chunk data
        chunks = []
        for segment in result.get("segments", []):
            chunks.append({
                "title_no": number,
                "title": title,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text")
            })

        # final structured output
        chunks_with_text = {
            "chunks": chunks,
            "text": result.get("text", "")
        }

        # save to JSON
        output_path = os.path.join("converted_texts", f"{audio}.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks_with_text, f, ensure_ascii=False, indent=2)
            print(f"saved transcription to {output_path}")
        except Exception as e:
            print(f"failed to save json for {audio}: {e}")

print("\nall done. check the converted_texts folder for the json files.")
