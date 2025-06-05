import ffmpeg
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav
from scipy.signal import wiener


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


def denoise_and_delay( noise_power, delay_ms, delay_gain, readFrom, writeTo):
   global _AUDIO_FILE_
   vid = ffmpeg.input(readFrom).video
   au = ffmpeg.input(readFrom).audio
   info = ffmpeg.probe(readFrom , cmd = "ffprobe")
   noChannels = info ["streams"][1]["channels"]
   au.output(_AUDIO_FILE_ , ac = noChannels).overwrite_output().run()
   sample_rate , sample_originals = wav.read(_AUDIO_FILE_)
   sample_float = sample_originals.astype(np.float64) #converting to float since we need higher percision 

   noise_reduction_streanght = max(3,min(15,int(abs(noise_power)/2)))

   denoised_audio = wiener (sample_float , mysize = noise_reduction_streanght)

   delay_samples = int ((delay_ms/1000) * sample_rate )
   delay_gain_decimal = delay_gain / 100.0

   delayed_signal = np.zeros_like(denoised_audio)
   if delay_samples < len(denoised_audio) :
       delayed_signal[delay_samples:] = denoised_audio[:-delay_samples]
       delayed_audio = denoised_audio + delayed_signal * delay_gain_decimal

   if np.max(np.abs(delayed_audio)) > 0:
       delayed_audio = delayed_audio * (32767 / np.max(np.abs(delayed_audio))*0.9)

   data2 = np.asarray(delayed_audio , dtype = np.int16)
   wav.write(_AUDIO_FILE_ , sample_rate , data2)
   ffmpeg.output(vid,ffmpeg.input(_AUDIO_FILE_) , writeTo).overwrite_output().run()
   return

def frameInterpolation (targetFps , readFrom , writeTo):
    stream = ffmpeg.input(readFrom)

    vid = stream.video 
    audio = stream.audio

    info = ffmpeg.probe(readFrom , cmd = "ffprobe")
    video_stream = next ((s for s in info['streams'] if s['codec_type'] == 'video' ), None)
    if 'avg_frame_rate' in video_stream:
        frame_rate_fraction = video_stream['avg_frame_rate']
        if '/' in frame_rate_fraction:
            num,den = map(int , frame_rate_fraction.split('/'))
            original_fps = num / den if den != 0 else 0
        else:
            original_fps = float(frame_rate_fraction)
    else:
        original_fps = 30

    if targetFps >= original_fps:
        interpolated = vid.filter('minterpolate' ,
                                  fps = targetFps,
                                  mi_mode = 'mci',
                                  mc_mode = 'aobmc',
                                  me_mode = 'bidir')
        ffmpeg.output(interpolated , audio , writeTo ,
                      vcodec = 'libx264',
                      acodec = 'aac').overwrite_output().run()
    else:
        decreased = vid.filter('fps',fps = targetFps)
        ffmpeg.output(decreased , audio , writeTo , vcodec = 'libx264' , acodec = 'aac').overwrite_output().run()
        


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



def makeCarLike(sideGain_db, filterOrder, readFrom, writeTo):
    global _AUDIO_FILE_
    
    # Extract video stream
    vid = ffmpeg.input(readFrom).video
    
    # Clean up existing audio file
    if os.path.exists(_AUDIO_FILE_):
        os.remove(_AUDIO_FILE_)
        
    # Convert sideGain from dB to linear
    sideGain_linear = 10 ** (sideGain_db / 20)
    
    # Calculate pan coefficients for stereo enhancement
    # In fact, for each output channel (new left or new right), we take 50% of the original left and 50% of the original right, 
    # then push them outward or inward depending on sideGain_linear.
    # This widens or narrows the stereo image.
    left_c0 = 0.5 + sideGain_linear * 0.5
    left_c1 = 0.5 - sideGain_linear * 0.5
    right_c0 = 0.5 - sideGain_linear * 0.5
    right_c1 = 0.5 + sideGain_linear * 0.5

    # Extract audio with stereo enhancement
    os.system(f'ffmpeg -i "{readFrom}" -af "pan=2c|c0={left_c0}*c0+{left_c1}*c1|c1={right_c0}*c0+{right_c1}*c1" {_AUDIO_FILE_}')
    
    # Read audio file. samples can be numpy array of shape (num_samples,) or is it is stereo it is (num_samples,2)
    # Each element is a signed 16-bit integer (int16) in the range [–32768, +32767].
    sample_rate, samples = wav.read(_AUDIO_FILE_)
    
    # Convert to float for processing(as lfilter works best with float numbers
    samples_float = samples.astype(np.float32) / 32768.0
    
    # Apply low-pass filter at 10000 Hz as specified
    # Butterworth design routines expect a cutoff in the range [0, 1], where 1 corresponds to Nyquist.
    nyquist = sample_rate / 2
    normalized_cutoff = 10000 / nyquist
    num, denom = butter(filterOrder, normalized_cutoff, btype="low")
    
    # Process each channel separately if stereo[Processes left channel separately: samples_float[:, 0],Processes right channel separately: samples_float[:, 1]]
    # For mono just apply the filter
    if len(samples_float.shape) == 2:
        filtered_samples = np.zeros_like(samples_float)
        for channel in range(samples_float.shape[1]):
            filtered_samples[:, channel] = lfilter(num, denom, samples_float[:, channel])
    else:
        filtered_samples = lfilter(num, denom, samples_float)
    
    
    # Convert back to int16 with proper scaling
    max_val = np.max(np.abs(filtered_samples))
    if max_val > 1.0:
        filtered_samples = filtered_samples / max_val
    
    # Final outputed sample which is then written back to the wav file
    output_samples = (filtered_samples * 32767).astype(np.int16)
    
    
    # Write processed audio back
    wav.write(_AUDIO_FILE_, sample_rate, output_samples)
    
    # Combine audio and video
    auInpF = ffmpeg.input(_AUDIO_FILE_)
    ffmpeg.output(vid, auInpF, writeTo).overwrite_output().run()
    
    return
        
    
