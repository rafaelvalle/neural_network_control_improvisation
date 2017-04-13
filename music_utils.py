from __future__ import division
import numpy as np
import pretty_midi as pm


def pianoroll_to_midi(pianoroll, fs, program=1, filepath='midifile.mid'):
    # create PrettyMIDI object
    midifile = pm.PrettyMIDI(resolution=fs, initial_tempo=60 / (4.0/fs))
    # create Instrument instance
    if type(program) is not int:
        program = pm.instrument_name_to_program(program)
    instrument = pm.Instrument(program=program)

    # cumbersomely slow, refactor
    for note in range(pianoroll.shape[0]):
        start = 0
        end = 0
        while end < pianoroll.shape[1]:
            # find where note starts
            while start < pianoroll.shape[1] and pianoroll[note, start] == 0:
                start += 1
            end = start + 1

            # nothing left
            if start == pianoroll.shape[1]:
                break

            # find where note ends
            while end < pianoroll.shape[1] and pianoroll[note, end] > 0:
                end += 1

            # add note to instrument
            instrument.notes.append(pm.Note(
                velocity=int(pianoroll[note, start]),
                pitch=note,
                start=start * 1.0/fs,
                end=end * 1.0/fs))
            start = end

    midifile.instruments.append(instrument)
    midifile.write(filepath)


