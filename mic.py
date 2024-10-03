import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io
import wave
import numpy as np
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tensorflow as tf
from tensorflow import keras

hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def audioPreprocessing(data):
    numSignals_max = 3
    target_len = 5000
    signal = np.zeros([numSignals_max, target_len], dtype=np.int32)
    #signal = np.zeros([numSignals_max, target_len])
    signalOffset = 5000
    peakOffset = -500
    signal_len = signalOffset + target_len + peakOffset + 10
    signal_flag = []

    x = data
    x_abs = np.abs(x)
    n_points = len(x_abs)
    threshold = 5000
    pos = 0
    
    for k in range(numSignals_max):
        if n_points > signal_len:
            for i in range(n_points):
                if i >= signalOffset:
                    if (x_abs[i] >= threshold):
                        pos = i
                        #print(pos)
                        start = pos + peakOffset
                        stop = start + target_len
                        signal[k,:] = x[start:stop]
                        #print(signal[k,:])
                        x = x[stop:]
                        x_abs = x_abs[stop:]
                        n_points = len(x)
                        pos = 0
                        signal_flag.append(1)
                        break                
    
    numSignals = len(signal_flag)
    sample = np.zeros([numSignals, target_len])
    signal = np.array(signal, dtype=np.float32)

    for i in range(numSignals):
        signal_abs = np.abs(signal[i,:])
        max_value = np.max(signal_abs)
        sample[i,:] = signal[i,:] / max_value
    
    return sample

st.title(":rainbow[Watermelon Sweetness Evaluation]")
model = tf.keras.models.load_model('wm_lstm.keras')

audio_bytes = audio_recorder(
    text="Click -->",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    #icon_name="user",
    #icon_size="4x",
    energy_threshold=(-1.0,1.0),
    pause_threshold=4.0,
    sample_rate=16000,
)
st.write("and then knock at a watermelon for 3 times")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    progress_text = "Operation in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)

    progress_bar.progress(20, text=progress_text)

    # Use io.BytesIO to handle the bytes as a file
    wav_io = io.BytesIO(audio_bytes)

    # Open the WAV file
    with wave.open(wav_io, 'rb') as wav_file:
        # Get the parameters of the WAV file
        params = wav_file.getparams()
        # Read the frames and convert to a NumPy array
        audio_data = np.frombuffer(wav_file.readframes(params.nframes), dtype=np.int16)

    sample = audioPreprocessing(audio_data)
    numSamples = sample.shape[0]
    #print(numSamples)
    progress_bar.progress(30, text=progress_text)

    if numSamples == 3:
        X_test = sample
        # Reshape input to be [samples, time steps, features]
        time_steps = 1
        X_test = X_test.reshape((X_test.shape[0], time_steps, X_test.shape[1]))
        Y_test = model.predict(X_test)
        #print(Y_test)
        progress_bar.progress(40, text=progress_text)

        max_index = np.argmax(Y_test, axis=1)
        #print(max_index)

        classNames = ['Less Sweet', 'Sweet']
        prediction_labels = []
        for i in range(numSamples):
            label = classNames[max_index[i]]
            prediction_labels.append(label)
        #print(prediction_labels)
        
        # Count occurrences of each class
        counts = np.bincount(max_index)
        #print(counts)
        max_pos = np.argmax(counts)
        #print(max_pos)
        result = classNames[max_pos]
        #print(result)
        progress_bar.progress(50, text=progress_text)

        with st.container(border=True):
            col1 = st.columns(2)
            with col1[0]:
                st.subheader(f":rainbow[Watermelon is {result}]")
                st.markdown(
                    """
                    <style>
                    .centered-image {
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.image("wm_256px.png", use_column_width=True, output_format="auto", width=None)
                progress_bar.progress(60, text=progress_text)
            with col1[1]:
                st.subheader(":green[Acoustic Signal]")
                st.line_chart(audio_data)
                progress_bar.progress(70, text=progress_text)
        
        st.subheader(":blue[Model Predictons]")
        
        with st.container(border=True):
            col2 = st.columns(numSamples)
            step = 70
            for i in range(numSamples):
                step = step + 10
                with col2[i]:
                    st.subheader(f":green[{prediction_labels[i]}]")
                    st.line_chart(sample[i,:])
                    progress_bar.progress(step, text=progress_text)
        
        #progress_bar.progress(100, text=progress_text)
        progress_bar.empty()
    else:
        st.subheader(":blue[Please try again]")