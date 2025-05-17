import ffmpeg
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav


import numpy as np
import os
_AUDIO_FILE_ = "audio.wav"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static")
pathMaker = lambda prefix, fileName: os.path.join(UPLOAD_FOLDER, f"{prefix}.{fileName.split('.')[-1]}")

def upscaler(tw, th, readFrom, writeTo):
    stream = ffmpeg.input(readFrom).video
    vid = stream.filter("scale", w=tw, h=th) 
    auInpf = ffmpeg.input(readFrom).audio
    ffmpeg.output(vid, auInpf, writeTo).overwrite_output().run()
    return 



def makePhoneLike(filterOrder, sideGain, readFrom, writeTo):
    global _AUDIO_FILE_
    vid = ffmpeg.input(readFrom).video
    
    if os.path.exists(_AUDIO_FILE_):
        os.remove(_AUDIO_FILE_)

    os.system(f'ffmpeg -i "{readFrom}" -af "pan=2c|c0={sideGain}*c0|c1={1-sideGain}*c1" {_AUDIO_FILE_}')
    sample_rate, samples_original = wav.read(_AUDIO_FILE_)
    num, denom = butter(filterOrder,  [800, 3400] , "bandpass", fs=sample_rate) 
    ot = lfilter(num, denom, samples_original)
    data2 = np.asarray(ot, dtype=np.int16) 
    wav.write(_AUDIO_FILE_, sample_rate, data2)
    auInpF = ffmpeg.input(_AUDIO_FILE_)
    ffmpeg.output(vid, auInpF, writeTo).overwrite_output().run()
    # ot = ffmpeg.output(prob, au, "video.mp4")
    return 

def denoise_and_delay(readFrom, writeTo, noise_power_db, delay_ms, delay_gain_percent):
    
    global _AUDIO_FILE_
    vid = ffmpeg.input(readFrom).video
    au = ffmpeg.input(readFrom).audio
    info = ffmpeg.probe(readFrom, cmd="ffprobe")
    
    noChannels = info["streams"][1]["channels"] 
    audioStream = au.output(_AUDIO_FILE_, ac=noChannels).overwrite_output().run()
    
    sample_rate, samples_original = wav.read(_AUDIO_FILE_)
    
    filter_order = 4
    low_cutoff = 300
    high_cutoff = 3400
    
    noise_factor = 10 ** (noise_power_db / 20)
    
    b, a = butter(filter_order, [low_cutoff, high_cutoff], btype='bandpass', fs=sample_rate)
    
    
    if len(samples_original.shape) > 1:  # For stereo audio
        denoised_audio = np.zeros_like(samples_original)
        for channel in range(samples_original.shape[1]):
            denoised_audio[:, channel] = lfilter(b, a, samples_original[:, channel])
    else:  # For mono audio
        denoised_audio = lfilter(b, a, samples_original)
    
    # reducing noise
    denoised_audio = denoised_audio * noise_factor
    
    
    delay_samples = int((delay_ms / 1000) * sample_rate)
    delay_gain = delay_gain_percent / 100
    
    
    if len(denoised_audio.shape) > 1:  # For stereo audio
        delayed_audio = np.zeros_like(denoised_audio)
        for channel in range(denoised_audio.shape[1]):
            delayed_channel = np.zeros_like(denoised_audio[:, channel])
            delayed_channel[delay_samples:] = denoised_audio[:-delay_samples, channel] if delay_samples < len(denoised_audio) else 0
            #scale the delayed audio        
            delayed_channel = delayed_channel * delay_gain
            # Mix with the original
            delayed_audio[:, channel] = denoised_audio[:, channel] + delayed_channel
    else:  # For mono audio
        delayed_audio = np.zeros_like(denoised_audio)
        delayed_audio[delay_samples:] = denoised_audio[:-delay_samples] if delay_samples < len(denoised_audio) else 0
        delayed_audio = delayed_audio * delay_gain
        delayed_audio = denoised_audio + delayed_audio
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(delayed_audio))
    if max_val > 32767:  # Max value for 16-bit audio
        delayed_audio = delayed_audio * (32767 / max_val)
    
    # Convert back to int16 for saving
    delayed_audio = np.asarray(delayed_audio, dtype=np.int16)
    
    # Save the processed audio
    wav.write(_AUDIO_FILE_, sample_rate, delayed_audio)
    
    # Combine processed audio with the original video
    auInpF = ffmpeg.input(_AUDIO_FILE_)
    ffmpeg.output(vid, auInpF, writeTo).overwrite_output().run()
    
    return




def applyGainCompression(threshold_db, limiter_db, readFrom, writeTo):
    
    stream = ffmpeg.input(readFrom)
    
    # Handle threshold parameter (ensure it's negative)
    abs_threshold = abs(threshold_db) if threshold_db < 0 else threshold_db
    threshold_point = f"-{abs_threshold}/-{abs_threshold}"
    
    # Handle limiter parameter
    mid_point = f"-{abs_threshold/2}/-{abs_threshold+limiter_db/2}"
    limiter_point = f"0/-{limiter_db}"
    
    # Combine points to create the compression curve
    points = f"{threshold_point}|{mid_point}|{limiter_point}"
    
    # Apply compression filter using 'compand'
    compressed_audio = stream.audio.filter(
        'compand',
        attacks='0.01',
        decays='0.5',
        points=points,
        gain='0'
    )
    
    # Combine original video with compressed audio
    result = ffmpeg.output(stream.video, compressed_audio, writeTo).overwrite_output().run()


def voiceEnhancement(preEmphasisAlpha, filterOrder, readFrom, writeTo):
 
    global _AUDIO_FILE_
    
    # Split video and audio
    vid = ffmpeg.input(readFrom).video
    au = ffmpeg.input(readFrom).audio
    
    # Extract audio to temporary file
    (
        ffmpeg
        .input(readFrom)
        .audio
        .output(_AUDIO_FILE_, acodec='pcm_s16le')
        .overwrite_output()
        .run()
    )
    
    # Read the audio file
    sample_rate, samples_original = wav.read(_AUDIO_FILE_)
    
    # Convert to mono if needed
    if len(samples_original.shape) > 1:
        samples_original = np.mean(samples_original, axis=1).astype(np.int16)
    
    # Apply pre-emphasis filter: y[n] = x[n] - α*x[n-1]
    alpha = min(max(float(preEmphasisAlpha) / 10, 0), 0.95)
    emphasized_samples = np.zeros_like(samples_original)
    emphasized_samples[0] = samples_original[0]
    emphasized_samples[1:] = samples_original[1:] - alpha * samples_original[:-1]
    
    # Apply band-pass filter (Butterworth)
    safe_filter_order = min(max(int(filterOrder), 1), 4)
    num, denom = butter(safe_filter_order, [800, 6000], "bandpass", fs=sample_rate)
    filtered_samples = lfilter(num, denom, emphasized_samples)
    
    # Normalize audio to avoid distortion
    if np.max(np.abs(filtered_samples)) > 0:
        filtered_samples = filtered_samples * (32767 / np.max(np.abs(filtered_samples)) * 0.9)
    
    # Convert to proper format
    enhanced_audio = np.asarray(filtered_samples, dtype=np.int16)
    
    # Write processed audio to temporary file
    wav.write(_AUDIO_FILE_, sample_rate, enhanced_audio)
    
    # Merge video with enhanced audio
    (
        ffmpeg
        .output(
            vid, 
            ffmpeg.input(_AUDIO_FILE_), 
            writeTo,
            vcodec='copy'  # Copy video to avoid re-encoding
        )
        .overwrite_output()
        .run()
    )
    
    # Clean up temporary file
    if os.path.exists(_AUDIO_FILE_):
        os.remove(_AUDIO_FILE_)
    
    return




def applyGrayscale(readFrom, writeTo):
    # Load video input
    stream = ffmpeg.input(readFrom)

    # Apply grayscale filter using 'format=gray'
    gray_stream = stream.video.filter("format", "gray")

    # Combine grayscale video with original audio
    result = ffmpeg.output(gray_stream,stream.audio, writeTo).overwrite_output().run()
    
    return



def colorInvert(readFrom, writeTo):
    input_video = ffmpeg.input(readFrom) # Get input streams
    inverted_video = input_video.video.filter('negate')  # Apply color inversion filter (negate) to the video stream 
    audio = input_video.audio    # Keep the original audio
    # Combine the inverted video with the original audio
    output = ffmpeg.output( 
        inverted_video, 
        audio, 
        writeTo,
        acodec='copy'  # Copy audio to avoid re-encoding
    )
    # Run the ffmpeg command
    output.overwrite_output().run()

    return

