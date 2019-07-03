EOH-2018: Sound visualization for electronic band EOH project.
================================================

Training Data Setup:
------------
```bash 
# NOTE: This takes a long time and the files generated are really large (4.5 Gb)
chmod +x philharmonia.sh && ./philharmonia.sh
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
