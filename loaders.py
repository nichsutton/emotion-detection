import os 
import pandas as pd

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
    filepath_df = pd.DataFrame(emotion_file_paths, columns=["Path"])
    ravdess_df = pd.concat([emotion_df, filepath_df], axis=1)

    ravdess_df.Emotions.replace({1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}, inplace=True)

    return ravdess_df

def load_crema_d (crema_d):
    cremad_dir = os.listdir(crema_d)

    emotion_files = []
    emotion_file_paths = []

    try:
        for audio_dir in cremad_dir:
            emotion_file_paths.append(crema_d + audio_dir)
            # Gets the "ANG, SAD, etc part of the filepath"
            part = audio_dir.split("_")[2]
            # Gets the first letter of the emotion
            audio_type = part[0].lower()

            if audio_type == "a":
                emotion_files.append("angry")
            elif audio_type == "d":
                emotion_files.append("disgust")
            elif audio_type == "f":
                emotion_files.append("fear")
            elif audio_type == "h":
                emotion_files.append("happy")
            elif audio_type == "n":
                emotion_files.append("neutral")
            elif audio_type + "a" == "sa":
                emotion_files.append("sad")
            else:
                emotion_files.append("surprised")
    except ValueError as identifier:
        print("Error:", identifier)

    emotion_df = pd.DataFrame(emotion_files, columns=["Emotions"])
    path_df = pd.DataFrame(emotion_file_paths, columns=["Path"])

    crema_d_df = pd.concat([emotion_df, path_df], axis=1)
    return crema_d_df




