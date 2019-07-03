EOH-2018: Sound visualization for electronic band EOH project.
================================================

Dependencies:
------------
* ffmpeg
* pygame
* pyaudio
* sklearn
* pickle
* wave

Training Data Setup:
------------
```bash 
# Install the philharmonia dataset, convert .mp3 files to .wav, and organize the directory.
# NOTE: This takes a long time to run and files created are very large (4.5 Gb). Only run this if interested in development.
chmod +x philharmonia.sh 
./philharmonia.sh
```

Machine Learning Models:
------------
```
# Musical note root harmonic detector:
python fund_note.py {int arg for num harmonics} {'overwrite' or 'keep' existing data} {'test' or 'finalize' models}
```

Demonstrations
-----------
```bash
# Piano synthesizer
python synth.py

# Sound visualization:
python sound_animation.py {int arg for num harmonics}

# FFT analysis of sound from microphone input:
python demo1.py {int arg for num harmonics}

# Musical notes visualization:
python draw.py {int arg for num harmonics}
```
