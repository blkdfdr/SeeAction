# /// script
# dependencies = [
#     "mss",
#     "keyboard",
#     "tensorflow==2.13.1",
#     "tensorflow-io-gcs-filesystem==0.30.0"
# ]
# ///

import os
import time
import threading
import datetime
from pathlib import Path

# Screen capture dependencies
try:
    import mss
    import numpy as np
    import cv2
    import keyboard
except Exception as e:
    print("Missing optional dependencies for recording: mss, opencv-python, numpy, keyboard")
    print("Install with: pip install mss opencv-python numpy keyboard")
    raise

# ML deps
# define tf up-front so we can check availability later
tf = None
try:
    import tensorflow as tf  # type: ignore
    from lib.net import VCModel
    import config
except Exception as e:
    print("TensorFlow or project imports failed: {}".format(e))
    # don't raise here because user might only want recording

# Configuration
DATA_DIR = config.data_dir
RECORD_FPS = 10
RECORD_REGION = None  # None means full screen
OUTPUT_VIDEOS_DIR = Path(DATA_DIR) / 'Video_Recorded'
OUTPUT_IMAGES_DIR = Path(DATA_DIR) / 'Images'
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

_recording = False


def record_screen(filename: str, stop_event: threading.Event, region=None, fps=10):
    """
    Records screen to `filename` until stop_event is set.
    Uses mss to capture the screen and cv2.VideoWriter to save.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1] if region is None else region
        width = monitor['width']
        height = monitor['height']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        print(f"Recording to {filename} ({width}x{height} @{fps}fps). Press Ctrl+Shift+R again to stop.")
        try:
            while not stop_event.is_set():
                img = np.array(sct.grab(monitor))
                # mss returns BGRA
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                out.write(frame)
                time.sleep(1.0 / fps)
        finally:
            out.release()
            print("Recording stopped and file closed.")


def extract_frames_from_video(video_path: str, output_dir: str, frame_rate: int = 5):
    """Extract frames using OpenCV into output_dir as five-digit PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate
    step = max(1, int(round(video_fps / frame_rate)))
    idx = 0
    out_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = os.path.join(output_dir, f"{out_idx:05d}.png")
            cv2.imwrite(out_path, frame)
            out_idx += 1
        idx += 1
    cap.release()
    print(f"Extracted {out_idx-1} frames to {output_dir}")
    return out_idx - 1


def load_model_and_labels():
    """Load VCModel and read label files (verbs, nouns, vocab). Returns model and lists."""
    m = None
    try:
        model = VCModel(is_training=False)
        model_path = Path(config.data_dir) / 'model' / (config.model_name + '_' + config.channel_name + '_' + config.val_name) / 'VCModel'
        try:
            model.load_weights(str(model_path))
            print("Model weights loaded from", model_path)
        except Exception as ex:
            print("Could not load weights:", ex)
        # label lists
        with open(os.path.join(config.data_dir, 'Labels', 'verbs.txt'), 'r') as f:
            verbs = [x.strip() for x in f.readlines()]
        with open(os.path.join(config.data_dir, 'Labels', 'nouns.txt'), 'r') as f:
            nouns = [x.strip() for x in f.readlines()]
        with open(os.path.join(config.data_dir, 'Labels', 'vocab.txt'), 'r') as f:
            vocab = [x.strip() for x in f.readlines()]
        return model, verbs, nouns, vocab
    except Exception as e:
        print("Failed to create or load model:", e)
        return None, [], [], []


def prepare_input_from_frames(images_dir):
    """Build a single sample entry (string format used by dataset) from extracted frames.
    This creates a single-line descriptor like: <video_name> <frame1> <frame2> ... <verb> <noun> <adv...>
    We leave labels as zeros because we're doing inference only.
    """
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
    if not files:
        raise RuntimeError("No frames found in " + images_dir)
    # choose up to 8 frames spaced evenly
    stride = 8
    if len(files) < stride:
        # duplicate frames if less than required
        chosen = [files[0]] + (files * (stride // len(files) + 1))[:stride-1]
    else:
        indices = np.linspace(0, len(files)-1, stride, dtype=int)
        chosen = [files[i] for i in indices]
    # build a pseudo "file_path" string used by dataset.process_label
    video_name = os.path.basename(images_dir)
    # append placeholder labels (verb, noun, adv tokens) - dataset will parse them but we won't use them
    placeholder = '0 ' * 7  # covers verb noun and 5-word adv (adjusted in dataset parsing)
    descriptor = video_name + ' ' + ' '.join(chosen) + ' ' + placeholder
    return descriptor


def run_inference_on_dir(images_dir):
    model, verbs, nouns, vocab = load_model_and_labels()
    if model is None:
        print("Model not available; skipping inference.")
        return
    # build tensor input using dataset utilities is more robust, but we'll re-create minimal preprocessing here
    # read images and match dataset.decode_img behaviour
    imgs = []
    for f in sorted([p for p in os.listdir(images_dir) if p.lower().endswith('.png')]):
        im = tf.io.read_file(os.path.join(images_dir, f))
        im = tf.io.decode_png(im, channels=3)
        im = tf.image.resize(tf.cast(im, tf.float32), [224, 224])
        imgs.append(im)
    if len(imgs) < 8:
        # pad
        while len(imgs) < 8:
            imgs.append(imgs[-1])
    imgs = tf.stack(imgs[:8], axis=0)
    # create fake crop and diff channels (duplicate image and zeros)
    crop = imgs
    diff = tf.zeros([8, 224, 224, 1], dtype=tf.float32)
    # concat into expected dict with batch dim
    # imgs: (8,224,224,3), crop: (8,224,224,3), diff: (8,224,224,1)
    # Concatenate along the channel axis to get (8,224,224,7) and add batch dim -> (1,8,224,224,7)
    image = tf.concat([imgs, crop, diff], axis=3)
    image = tf.expand_dims(image, 0)
    try:
        outputs = model({'image': image})
        # outputs may be dict
        if isinstance(outputs, dict):
            if 'verb' in outputs:
                verb_logits = outputs['verb']
                verb_idx = int(tf.argmax(verb_logits, axis=1).numpy()[0])
                print("Predicted verb:", verbs[verb_idx] if verbs and verb_idx < len(verbs) else verb_idx)
            if 'noun' in outputs:
                noun_logits = outputs['noun']
                noun_idx = int(tf.argmax(noun_logits, axis=1).numpy()[0])
                print("Predicted noun:", nouns[noun_idx] if nouns and noun_idx < len(nouns) else noun_idx)
            if 'adv' in outputs:
                adv_logits = outputs['adv']
                # adv logits shape: (batch, seq_len, vocab)
                seq = []
                for i in range(adv_logits.shape[1]):
                    tok = int(tf.argmax(adv_logits[0, i], axis=-1).numpy())
                    seq.append(vocab[tok] if tok < len(vocab) else str(tok))
                print("Predicted adv sequence:", ' '.join(seq))
        else:
            print("Model returned unexpected output type:", type(outputs))
    except Exception as e:
        print("Error running inference:", e)


def on_hotkey():
    global _recording
    if not _recording:
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_video = os.path.join(OUTPUT_VIDEOS_DIR, f'record_{now}.mp4')
        stop_event = threading.Event()
        t = threading.Thread(target=record_screen, args=(out_video, stop_event, RECORD_REGION, RECORD_FPS), daemon=True)
        _recording = (out_video, stop_event, t)
        t.start()
    else:
        out_video, stop_event, t = _recording
        stop_event.set()
        t.join()
        _recording = False
        # extract frames
        video_basename = Path(out_video).stem
        images_output = os.path.join(OUTPUT_IMAGES_DIR, video_basename)
        extract_frames_from_video(out_video, images_output, frame_rate=5)
        # run inference
        run_inference_on_dir(images_output)


def main():
    print("Press Ctrl+Shift+R to start/stop recording. Press ESC to quit.")
    keyboard.add_hotkey('ctrl+shift+r', lambda: on_hotkey())
    keyboard.wait('esc')


if __name__ == '__main__':
    main()
