EOH-2018: Sound visualization for EOH electronic band
================================================
This is an old project presented at UIUC for EOH-2018 that was last maintained by mlchrln2. Some demonstrations are included to get an idea of where this project was going. The rest of this project is private and has moved over to mlchrln2's GitLab account @SimplySymplectic88. Another spin off project is in the making in Github named BeyondMIDI. Check there for more updates on this project.

Dependencies:
------------
* ffmpeg
* pygame
* pyaudio
* sklearn
* pickle
* wave

Setup:
------------
```bash 
# Install the philharmonia dataset, convert .mp3 files to .wav, and organize the directory.
# NOTE: This takes a long time to run and storage space required is around 4.5 Gb.
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
python3 synth.py

# Sound visualization (args: numHarmonics 1-20):
python3 sound_animation.py 3

# FFT analysis of sound from microphone input (args: numHarmonics 1-20):
python3 demo1.py 3

# Musical notes visualization (args: numHarmonics 1-20):
python3 draw.py 3```
