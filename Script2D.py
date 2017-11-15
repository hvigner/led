
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import wave
import scipy.signal as sc
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import pyaudio
from PIL import Image
 


# In[16]:


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


# In[20]:



##STATE EQUATIONS


def waveform(signal):
    if(json_data["state"] == 1):
        scale_down = int(json_data["horScale"])
        shorten = int(json_data["verScale"])
        signal = signal[0:len(signal)/scale_down]
        fs = spf.getframerate()/scale_down
        signal[signal > np.max(np.abs(signal))/shorten] = np.max(signal)/shorten
        signal[signal < -np.max(np.abs(signal))/shorten] = -np.max(np.abs(signal))/shorten




        Time=np.linspace(0, len(signal)/fs, num=len(signal))


        fig =plt.figure(1)
        plt.title('Signal Wave...')
        time_domain = plt.plot(Time,signal)

        fig.add_subplot(111)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()
        im = fig2img(fig)
        im.show()

        # Now we can save it to a numpy array.
        #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.show()




def frequency_domain(copy):
    if(json_data["state"] == 0):

        fs = 10e-6

        f, t, Sxx = sc.spectrogram( copy,fs=1.0, window=('tukey', 0.25))
        plt.pcolormesh(t, f, Sxx)
        fig =plt.figure(1)
        fig.add_subplot(111)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()
        fig2img(fig)
        # Now we can save it to a numpy array.
        #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')
        #plt.show()
     
        
        
        
        
 



    


# In[21]:


from scipy.signal import butter, lfilter
import json
##Butterpass
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    


# In[23]:




#AUDIO INPUT

FORMAT = pyaudio.paInt16

CHANNELS = 1

RATE = 44100

CHUNK = 1024

RECORD_SECONDS = 10

WAVE_OUTPUT_FILENAME = "output.wav"

lowcut = 200.0
highcut = 4000.0

audio = pyaudio.PyAudio()

 

    # start Recording

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
while(1):
    json_data = json.load(open('fsm.json'))
    print "recording"

    frames = []


    # start the stream (4)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

        data = stream.read(CHUNK)

        frames.append(data)

    print "finished recording"


    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

    waveFile.setnchannels(CHANNELS)

    waveFile.setsampwidth(audio.get_sample_size(FORMAT))

    waveFile.setframerate(RATE)

    waveFile.writeframes(b''.join(frames))

    waveFile.close()
    
    spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    signal = butter_bandpass_filter(signal,lowcut,highcut,RATE)
    #f = open("fsm.txt","r")
   
    state = json_data["state"]
    #print state
    if state == []:
        continue
    copy= signal.copy()
    waveform(signal)
    frequency_domain(copy)
    


 # stop Recording

stream.stop_stream()
    
stream.close()
audio.terminate()

