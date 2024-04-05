import sys
import os 
import pandas as pd

# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
#
# Vocal channel (01 = speech, 02 = song).
#
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
#
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
#
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
#
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
#
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

def load_ravdess(ravdess):
    rav_dir = os.listdir(ravdess)

    emotion_files = []
    emotion_file_paths = []

# find each actor file in the dataset
    try:
        for actor_dir in rav_dir:
            actor = os.listdir(ravdess + actor_dir)
            for audio in actor:
                section = audio.split(".")[0]
                section = section.split("-")
                # add the emotion 
                emotion_files.append(int(section[2]))
                emotion_file_paths.append(ravdess + actor_dir + "/" + audio)
    except:
        pass

    emotion_df = pd.DataFrame(emotion_files, columns=["Emotions"])
    filepath_df = pd.DataFrame(emotion_file_paths, columns=["Pathname"])
    ravdess_df = pd.concat([emotion_df, filepath_df], axis=1)

    ravdess_df.Emotions.replace({1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}, inplace=True)

    return ravdess_df

sys.modules[__name__] = load_ravdess


