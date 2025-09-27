import os
import numpy as np
from PIL import Image
import cv2

import soundfile as sf

# This is a workaround to make the example runnable without installing the package.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modality_processor.processor import ModalityProcessor
from modality_processor.modules.image_module import ImageModule
from modality_processor.modules.video_module import VideoModule
from modality_processor.modules.text_module import TextModule
from modality_processor.modules.audio_module import AudioModule
from modality_processor.config import ImageModuleConfig, VideoModuleConfig, TextModuleConfig, AudioModuleConfig

def create_dummy_files():
    """Creates dummy files for demonstration purposes."""
    print("Creating dummy files for testing...")
    # Create a dummy image
    img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save("dummy_image.png")
    print("- dummy_image.png created.")

    # Create a dummy text file
    with open("dummy_text.txt", "w") as f:
        f.write("This is a test of the modality processor system.\n")
        f.write("It should handle text, image, and video data streams.\n")
    print("- dummy_text.txt created.")

    # Create a dummy video file (requires opencv)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('dummy_video.mp4', fourcc, 1, (64, 64))
    for _ in range(10): # 10 frames
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        video.write(frame)
    video.release()
    print("- dummy_video.mp4 created.")

    # Create a dummy audio file (requires soundfile)
    samplerate = 44100  # samples per second
    duration = 1.0  # seconds
    frequency = 440.0  # Hz (A4 note)
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)
    sf.write('dummy_audio.wav', data.astype(np.int16), samplerate)
    print("- dummy_audio.wav created.")
    print("-" * 20)


def main():
    """
    Main function to demonstrate the ModalityProcessor with policy management.
    """
    create_dummy_files()

    processor = ModalityProcessor()

    # Register all modules
    processor.register_module(ImageModule, ImageModuleConfig())
    processor.register_module(TextModule, TextModuleConfig())
    processor.register_module(VideoModule, VideoModuleConfig(extract_audio=True))
    processor.register_module(AudioModule, AudioModuleConfig())

    # --- DEMO 1: Process video with only 'video' modality enabled ---
    print("\n--- DEMO 1: Processing video with policy {'enabled_modalities': ['video']} ---")
    processor.set_policy({'enabled_modalities': ['video', 'image', 'text']}) # Audio is disabled
    
    packet_stream = processor.process("dummy_video.mp4")
    
    # Consume the generator into a list to count packets
    all_packets_demo1 = list(packet_stream)
    video_packets = [p for p in all_packets_demo1 if p.modality == 'video']
    audio_packets = [p for p in all_packets_demo1 if p.modality == 'audio']
    
    print(f"  -> Found {len(video_packets)} video packets.")
    print(f"  -> Found {len(audio_packets)} audio packets.")
    print("-" * 20)

    # --- DEMO 2: Process video with 'video' and 'audio' modalities enabled ---
    print("\n--- DEMO 2: Processing video with policy {'enabled_modalities': ['video', 'audio']} ---")
    processor.set_policy({'enabled_modalities': ['video', 'audio', 'image', 'text']}) # All enabled
    
    packet_stream_all = processor.process("dummy_video.mp4")
    
    # Consume the generator to count packets
    all_packets_demo2 = list(packet_stream_all)
    video_packets_all = [p for p in all_packets_demo2 if p.modality == 'video']
    audio_packets_all = [p for p in all_packets_demo2 if p.modality == 'audio']
    
    print(f"  -> Found {len(video_packets_all)} video packets.")
    print(f"  -> Found {len(audio_packets_all)} audio packets.")
    
    # Print details of a few packets to show they are being generated
    print("\n  --- Sample Packets ---")
    for i, packet in enumerate(all_packets_demo2):
        if i >= 5:
            print("    ...")
            break
        print(f"    Packet {i}: Modality={packet.modality}, Timestamp={packet.timestamp:.2f}, Shape={packet.data.shape}")
    print("-" * 20)


    # Clean up
    print("\nCleaning up dummy files...")
    dummy_files = ["dummy_image.png", "dummy_text.txt", "dummy_video.mp4", "dummy_audio.wav"]
    for file_path in dummy_files:
        os.remove(file_path)
    print("Done.")


if __name__ == "__main__":
    main()

