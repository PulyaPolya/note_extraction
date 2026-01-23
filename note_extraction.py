import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from random import randrange
from argparse import ArgumentParser
from tqdm import tqdm
import os
import shutil

def init_parser():
    parser = ArgumentParser(description='Note extraction')
    parser.add_argument("--top_db", type=float, default=40, help="Top db parameter that defines threshold when the note is still considered to be played")
    parser.add_argument("--input_folder",  type=str, default="audio", help="Input folder that contains audio")
    parser.add_argument("--audio_file",  type=str, default="heckelphone.wav", help="Audio file from which the notes should be extracted")
    parser.add_argument("--min_duration", type=float, default=0.7, help="Minimum duration of each played note")
    parser.add_argument("--loudness", type=bool, default=False, help="If adding the estimated loudness to the file name is desired")
    parser.add_argument("--output_folder", type=str, default="results", help="Output folder name")
    parser.add_argument("--instrument", type=str, default="Heckelphone", help="Name of the instrument")

    return parser

def plot_all_notes(y_trimmed):
    frame_length = 2048  # length of each frame for analysis
    hop_length = 512     # number of samples between frames

    rms = librosa.feature.rms(y=y_trimmed, frame_length=frame_length, hop_length=hop_length)[0]
    amplitude = np.abs(y_trimmed)

    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    # adjust height and distance as needed
    peaks, _ = find_peaks(rms, height=0.01, distance=200)  

    num_peaks = len(peaks)

    plt.figure(figsize=(20, 6))
    librosa.display.waveshow(y_trimmed, sr=sr, alpha=0.5)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    for i, (start, end) in enumerate(filtered_intervals):
        plt.vlines(start/ sr, -0.03, 0.03, colors = 'green',  label='Start of note' if i == 0 else "")  # green lines mark the start of the extracted notes
        plt.vlines(end/ sr, -0.03, 0.03, colors = 'red', label='End of note' if i == 0 else "")   # red lines mark the ending of the notes
    plt.legend(loc='upper right')
    plt.xlim(0, len(y_trimmed) / sr)
    plt.tight_layout()
    plt.savefig(f"figures/All_notes.pdf")
    #plt.show()

def plot_random_note(filtered_intervals):
    num_note =randrange(len(filtered_intervals))
    plt.plot(notes[num_note])  # plot the waveform of the extracted note
    plt.title(f"Waveform of extracted note #{num_note}")
    plt.savefig(f"figures/Note_{num_note}.pdf")

def trim_silence(y, top_db = 40):
    y_trimmed, idx = librosa.effects.trim(
    y,
    top_db=top_db  # threshold in dB below reference
)
    # alternatively, beginning of the track can be manually trimmed 
    #y_trimmed = y[10*sr:] # removing noise in the beginning that lasted approximately 10 seconds
    return y_trimmed

def concatenate_with_clicks(notes):
    click_duration =1  
    click_freq = 1000  
    click_sr = 22050 
    # adding clicks to separate notes
    t = np.linspace(0, click_duration, int(click_sr * click_duration), endpoint=False)
    click = 0.1 * np.sin(2 * np.pi * click_freq * t) 
    concatenated_audio = notes[0]
    for note in notes[1:]:
        concatenated_audio = np.concatenate((concatenated_audio, click, note))
    return concatenated_audio

def detect_note(note,sr = 48000):

    # get the fundamental frequency f0
    f0 = librosa.yin(
        note,
        sr=sr,
        fmin=librosa.note_to_hz("C2"),  
        fmax=librosa.note_to_hz("C7"),  
    )
    # remove frames with no f0
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) == 0:
        raise ValueError("No valid f0 detected. Maybe too noisy or silence?")

    # use median f0 as an estimate
    f0_median = np.median(f0_valid)

    # convert frequency to note name
    note_name = librosa.hz_to_note(f0_median)

    return note_name

def estimate_loudness_db(note):
    # RMS amplitude
    rms = np.sqrt(np.mean(note**2))
    
    # convert to dB
    rms_db = 20 * np.log10(rms + 1e-12)
    return rms_db

def loudness_to_dynamic(rms_db, boundaries):
    # converting loudness to dynamics based on boundaries
    if rms_db < boundaries[0]:
        return "p"   
    elif rms_db < boundaries[1]:
        return "mp"  
    elif rms_db < boundaries[2]:
        return "mf"  
    else:
        return "f"  
    
def get_loudness_percentiles(notes):
    loudness_values = []
    for i in range(len(notes)):
        loudness = estimate_loudness_db(notes[i])
        loudness_values.append(loudness)
    
    # getting quartiles based on loudness and associating them with dynamics
    p_mp = np.percentile(loudness_values, 25)
    mp_mf = np.percentile(loudness_values, 50)
    mf_f = np.percentile(loudness_values, 75)
    return p_mp, mp_mf, mf_f

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    y, sr = librosa.load(os.path.join(args.input_folder, args.audio_file), sr=None)
    # removing noise in the beginning that lasted approximately 10 seconds
    y_trimmed = trim_silence(y, top_db=args.top_db)
    intervals = librosa.effects.split(y_trimmed, top_db=args.top_db)  # getting individual notes, parameter top_db defines how sensitive the cut is
    print(f"Number of intervals is {len(intervals)}")
    # define the minimum duration of the note
    min_note_duration =  sr*args.min_duration
    diff =  [intervals[i][1] - intervals[i][0] for i in range(len(intervals))]  # calculating the time difference between extracted notes
    filtered_intervals =  [interval for i, interval in enumerate(intervals) if diff[i] >= min_note_duration] # removing noise that last less a thershold
    print(f"Number of extracted notes after filtering is {len(filtered_intervals)}")

    # increasing the note intervals to capture the attacks and decays properly
    start_addition = int(0.1*sr)
    end_addition =  int(0.2*sr)
    notes = []                                           
    for i, (start, end) in enumerate(filtered_intervals):
        # adding little time the beginning for attack 
        note = y_trimmed[start-start_addition:end+ end_addition]  
        notes.append(note)

    os.makedirs("figures", exist_ok=True)
    # plotting a random note
    plot_random_note(filtered_intervals)
    # plotting all notes together
    plot_all_notes(y_trimmed)
    
    # estimating relative loudness of notes within the given piece
    if args.loudness:
        p_mp, mp_mf, mf_f = get_loudness_percentiles(notes)
    
    # saving the notes
    os.makedirs(args.output_folder, exist_ok=True)
    for i, note in tqdm(enumerate(notes)):
        note_name = detect_note(note, sr)
        dynamics = loudness_to_dynamic(estimate_loudness_db(note), boundaries=(p_mp, mp_mf, mf_f)) if args.loudness else "mf"
        wav_file_path = os.path.join(args.output_folder, f"{args.instrument}_{dynamics}_{note_name}.wav")
        sf.write(wav_file_path,note, sr)

    # saving concatenated notes in one file as a sanity check
    concatenated_audio = concatenate_with_clicks(notes)
    wav_file =f"{args.output_folder}\\concatenated_result.wav"    
    sf.write(wav_file,concatenated_audio, sr)
    shutil.make_archive("results", 'zip',"results") 

