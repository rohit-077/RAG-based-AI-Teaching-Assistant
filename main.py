import subprocess
import sys


# Helper function to run any Python script safely
def run_script(script_name):
    """runs a python script and stops if anything goes wrong."""
    print(f"\n----------------------------------------")
    print(f"running: {script_name}")
    print("----------------------------------------")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True
        )
        print(f"\n{script_name} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nerror: {script_name} failed with exit code {e.returncode}. stopping here.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nerror: couldn't find {script_name}. make sure it exists in this folder.")
        sys.exit(1)
    except Exception as e:
        print(f"\nerror while running {script_name}: {e}")
        sys.exit(1)


def main():
    print("starting the full ML tutorial processing pipeline...")
    print("make sure your ollama and ffmpeg services are up and running before starting.\n")

    # step 1: convert videos to audio
    run_script("video_to_audio.py")

    # step 2: transcribe audios into text chunks using whisper
    run_script("create_chunks.py")

    # step 3: create embeddings for all chunks
    run_script("create_embeddings.py")

    # step 4: process user query and build final prompt
    run_script("process_query.py")

    # step 5: generate final response using deepseek model
    run_script("generate_response.py")

    print("\n----------------------------------------")
    print("pipeline completed successfully.")
    print("check your 'outputs/answer.txt' file for the final response.")
    print("----------------------------------------\n")


if __name__ == "__main__":
    main()
