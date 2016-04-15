# neural_network_control_improvisation
Experiments on learning controllers and mining specifications using neural networks
1) Hard specs
    # Learning how to play notes in a pitch set
    a) harmonic specifications given pitch class set, with and without common
    pitches

2) Hard specs
    # Learning how to play notes in a pitch set
    a) harmonic specifications given pitch class set
    c) (in chord, not in chord)

3) Hard specs, with implictly t-1 dependency
    # Learn how to play notes in the pitchset respecting contour
    a) harmonic specifications given pitch class set
    b) intervallic specifications up, keep, down, (1,1,1,0,0,0,-1,-1,-1)

4) Hard specs, with explicit t-1 dependency
    # Learn how to play notes in the pitch set respecting contour and harmony
    a) harmonic specifications given pitch class set
    b) intervallic specifications up, keep, down, (1,1,1,0,0,0,-1,-1,-1)
    c) (in chord, not in chord)*

5) Hard and soft specs, with implicit t-1 dependency
    # Learn how to play notes in the pitch set respecting contour distribution
    # and harmony
    a) harmonic specifications given pitch set within an octave
    b) intervallic specifications up, keep, down, P(i|t)

6) Hard and soft specs, with explicit t-1 dependency
    a) harmonic specifications given pitch set within an octave
    b) intervallic specifications up, keep, down, P(i|t)
    c) (in chord, not in chord)*

7) Learn a chord progression associated with a melody 
