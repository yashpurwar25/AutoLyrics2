import os
from datasets import load_dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm

def download_and_save(dataset_name, subset=None, split="train", output_folder="data/raw"):
    print(f"🚀 Starting ingestion for: {dataset_name}")
    
    try:
        # Using streaming=True to avoid downloading 10GB of data at once
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    except Exception as e:
        print(f"❌ Hugging Face Error: {e}")
        return

    clean_name = dataset_name.replace("/", "_").replace(" ", "_")
    base_path = os.path.join(output_folder, clean_name)
    audio_path = os.path.join(base_path, "audio")
    lyric_path = os.path.join(base_path, "lyrics")
    
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(lyric_path, exist_ok=True)

    print(f"📁 Saving samples to: {base_path}")

    count = 0
    # We take 50 samples for the smoke test to keep it fast
    for i, example in enumerate(tqdm(ds, desc="Downloading")):
        if i >= 50: break 
        try:
            if 'audio' in example:
                audio_data = example['audio']
                audio_array = audio_data['array']
                sample_rate = audio_data['sampling_rate']
            else:
                continue

            # Google FLEURS uses 'transcription' as the key
            text = None
            for key in ['transcription', 'text', 'lyrics', 'sentence']:
                if key in example and example[key]:
                    text = example[key]
                    break
            
            if text is None: continue

            filename = f"sample_{i}"
            sf.write(os.path.join(audio_path, f"{filename}.wav"), audio_array, sample_rate)
            with open(os.path.join(lyric_path, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write(str(text))
            
            count += 1
        except Exception:
            continue
            
    print(f"\n✅ DONE! Successfully saved {count} samples to {base_path}")

if __name__ == "__main__":
    # Using google/fleurs because it is PUBLIC and requires no permission.
    TARGET_DATASET = "google/fleurs" 
    
    try:
        # FLEURS requires a language subset. 'en_us' is English US.
        download_and_save(TARGET_DATASET, subset="en_us", split="train")
    except Exception as e:
        print(f"Fatal Error during ingestion: {e}")