from __future__ import division
import numpy as np
import pretty_midi as pm
import pdb


def interpolate_between_beats(beats, n_steps):
    interpolations = np.array([
        np.linspace(beats[i], beats[i+1], n_steps+1)[:-1]
        for i in range(len(beats)-1)]).flatten()
    return interpolations


def quantize(data, times):
    # in place
    for instrument in data.instruments:
        for note in instrument.notes:
            i = max(0, np.argmax(note.start < times))
            if abs(note.start - times[i-1]) < abs(note.start - times[i]):
                note.start = times[i-1]
            else:
                note.start = times[i]
            k = max(0, np.argmax(note.end < times))
            if abs(note.end - times[k-1]) < abs(note.end - times[k]):
                note.end = times[k-1]
            else:
                note.end = times[k]


def pianoroll_to_midi(pianoroll, fs, program=1, filepath='midifile.mid',
                      scale=True):
    # create PrettyMIDI object
    midifile = pm.PrettyMIDI(resolution=fs, initial_tempo=60 / (4.0/fs))
    # create Instrument instance
    if type(program) is not int:
        program = pm.instrument_name_to_program(program)
    instrument = pm.Instrument(program=program)
    # scale piano roll to [0, 127]
    if scale:
        pianoroll -= pianoroll.min()
        pianoroll = 127*(pianoroll / pianoroll.max())
        pianoroll = pianoroll.astype(int)
    # cumbersomely slow, refactor
    for note in range(pianoroll.shape[0]):
        start = 0
        end = 0
        while end < pianoroll.shape[1]:
            # find where note starts
            while start < pianoroll.shape[1] and pianoroll[note, start] == 0:
                start += 1
            if start == pianoroll.shape[1]:
                break

            # find where note ends
            end = start + 1
            while end < pianoroll.shape[1] and pianoroll[note, end] > 0 and pianoroll[note, end-1] == pianoroll[note, end]:
                end += 1

            # add note to instrument
            instrument.notes.append(pm.Note(
                velocity=pianoroll[note, start],
                pitch=note,
                start=start * 1.0/fs,
                end=end * 1.0/fs))
            start = end

    midifile.instruments.append(instrument)
    midifile.write(filepath)
