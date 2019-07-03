# EOH 2018

Sound visualization for electronic band EOH project.

run this before everything else:
  NOTE: this takes a long time and the files generated are really large (4.5 Gb)
    chmod +x philharmonia.sh
    ./philharmonia.sh

demos:
  piano synth:
  #for keyboard to piano sound mapping run:
    python synth.py

sound visuals:
  for visualization on sound waves and its components run:
    python sound_animation.py {int arg for num harmonics}

for simple fft analysis of sound files or microphone input run:
  python demo1.py {int arg for num harmonics}

note visuals:
  for drawing notes on a staff run:
    python draw.py {int arg for num harmonics}



fine tuning of models:
  fundamental detection:
    for creating ML models for picking the fundamental frequency run:
      python fund_note.py {int arg for num harmonics} {'overwrite' or 'keep' existing data} {'test' or 'finalize' models}
