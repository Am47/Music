import ffmpeg
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav


import numpy as np
# from pydub import AudioSegment
import os
_AUDIO_FILE_ = "audio.wav"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', "result.mp4")


def upscaler(tw, th, readFrom, writeTo):
    stream = ffmpeg.input(readFrom).video
    vid = stream.filter("scale", w=tw, h=th) 
    auInpf = ffmpeg.input(readFrom).audio
    ffmpeg.output(vid, auInpf, writeTo).overwrite_output().run()
    return 



def makePhoneLike(filterOrder, sideGain, readFrom, writeTo):
    global _AUDIO_FILE_
    vid = ffmpeg.input(readFrom).video
    au = ffmpeg.input(readFrom).audio
    info = ffmpeg.probe(readFrom, cmd="ffprobe") # metadata of the file! 
    noChannels = info["streams"][1]["channels"] # extract number of channels from original file
    audioStream = au.output(_AUDIO_FILE_, ac=(noChannels if sideGain else 1)).overwrite_output().run()
    sample_rate, samples_original = wav.read(_AUDIO_FILE_)
    num, denom = butter(filterOrder,  [800, 3400] , "bandpass", fs=sample_rate) 
    ot = lfilter(num, denom, samples_original)
    data2 = np.asarray(ot, dtype=np.int16) 
    wav.write(_AUDIO_FILE_, sample_rate, data2)
    auInpF = ffmpeg.input(_AUDIO_FILE_)
    ffmpeg.output(vid, auInpF, writeTo).overwrite_output().run()
    # ot = ffmpeg.output(prob, au, "video.mp4")
    return 


def denoise_and_delay(filename, noise_power_db, delay_ms, delay_gain_percent):
    """
    Apply denoise using Wiener filter and delay effects to audio.
    
    Parameters:
    - filename: Input video file path
    - noise_power_db: Noise power in dB (default: -15)
    - delay_ms: Delay time in milliseconds (default: 100)
    - delay_gain_percent: Delay gain as percentage (default: 50)
    """
    global _AUDIO_FILE_
    vid = ffmpeg.input(filename).video
    au = ffmpeg.input(filename).audio
    info = ffmpeg.probe(filename, cmd="ffprobe")
    
    # Extract audio to separate file
    noChannels = info["streams"][1]["channels"] 
    audioStream = au.output(_AUDIO_FILE_, ac=noChannels).overwrite_output().run()
    
    # Read the audio file
    sample_rate, samples_original = wav.read(_AUDIO_FILE_)
    
    # Step 1: Denoise the audio using a Wiener filter
    from scipy.signal import wiener
    
    # Convert noise power from dB to linear scale for adjusting filter strength
    # More negative dB values will result in stronger noise reduction
    noise_reduction_strength = int(abs(noise_power_db)) + 1  # Use as mysize parameter
    
    # Apply Wiener filter for denoising
    if len(samples_original.shape) > 1:  # For stereo audio
        denoised_audio = np.zeros_like(samples_original, dtype=np.float64)
        for channel in range(samples_original.shape[1]):
            # Apply Wiener filter with adaptive noise reduction strength
            denoised_audio[:, channel] = wiener(samples_original[:, channel], 
                                               mysize=noise_reduction_strength)
    else:  # For mono audio
        denoised_audio = wiener(samples_original, mysize=noise_reduction_strength)
    
    # Step 2: Add delay effect (digital delay implementation)
    # Convert delay parameters
    delay_samples = int((delay_ms / 1000) * sample_rate)
    delay_gain = delay_gain_percent / 100
    
    # Create a delayed version of the audio
    if len(denoised_audio.shape) > 1:  # For stereo audio
        delayed_audio = np.zeros_like(denoised_audio)
        for channel in range(denoised_audio.shape[1]):
            # Create delayed signal
            delayed_signal = np.zeros_like(denoised_audio[:, channel])
            delayed_signal[delay_samples:] = denoised_audio[:-delay_samples, channel] if delay_samples < len(denoised_audio) else 0
            
            # Mix original and delayed signal
            delayed_audio[:, channel] = denoised_audio[:, channel] + delayed_signal * delay_gain
    else:  # For mono audio
        # Create delayed signal
        delayed_signal = np.zeros_like(denoised_audio)
        delayed_signal[delay_samples:] = denoised_audio[:-delay_samples] if delay_samples < len(denoised_audio) else 0
        
        # Mix original and delayed signal
        delayed_audio = denoised_audio + delayed_signal * delay_gain
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(delayed_audio))
    if max_val > 0:
        scale_factor = 32767 / max_val  # Scale to maximum 16-bit value
        delayed_audio = delayed_audio * scale_factor
    
    # Convert back to int16 for saving
    delayed_audio = np.asarray(delayed_audio, dtype=np.int16)
    
    # Save the processed audio
    wav.write(_AUDIO_FILE_, sample_rate, delayed_audio)
    
    # Combine processed audio with the original video
    auInpF = ffmpeg.input(_AUDIO_FILE_)
    ffmpeg.output(vid, auInpF, UPLOAD_FOLDER).overwrite_output().run()
    
    return





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
        .run(quiet=True)
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
    out, err = output.overwrite_output().run(capture_stdout=True, capture_stderr=True)
    print("FFmpeg stdout:", out)
    print("FFmpeg stderr:", err)

    return

def applyGainCompression():
    return

def voiceEhancement():
    return

