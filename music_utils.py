from __future__ import division
import numpy as np


def generateNot1(seq, spec, offset, ratio=0.5, as_proll=False):
    if as_proll:
        ts_nt = np.argwhere(seq == 1)
        viol_ts_nt = ts_nt[np.random.choice(np.arange(len(ts_nt)),
                                            int(np.sum(seq > 0) * ratio),
                                            replace=False)]
        seq_viol = np.copy(seq)

        i = -1
        for i in xrange(len(spec)-1):
            # can be optimized by keeping track of indices visited...
            viol_ts_i = [j[0] for j in viol_ts_nt
                         if j[0] >= spec[i][1] and j[0] < spec[i+1][1]]
            seq_viol[viol_ts_i, np.random.choice(
                [x+offset for x in xrange(0, 12) if x not in spec[i][0]],
                len(viol_ts_i))] = 1

        viol_ts_i = [j[0] for j in viol_ts_nt if j[0] >= spec[-1][1]]
        seq_viol[viol_ts_i, np.random.choice(
            [x+offset for x in xrange(0, 12) if x not in spec[-1][0]],
            len(viol_ts_i))] = 1

        # set modified notes to zero
        seq_viol[viol_ts_nt[:, 0], viol_ts_nt[:, 1]] = 0
    else:
        note_ids = np.where(seq >= 0)[0]
        viol_ids = np.random.choice(note_ids, int(np.sum(seq > 0) * ratio),
                                    replace=False)
        seq_viol = np.copy(seq)
        i = -1

        for i in xrange(len(spec)-1):
            viol_ids_i = [j for j in viol_ids
                          if j >= spec[i][1] and j < spec[i+1][1]]
            seq_viol[viol_ids_i] = np.random.choice(
                [x+offset for x in xrange(0, 12) if x not in spec[i][0]],
                len(viol_ids_i))

        viol_ids_i = [j for j in viol_ids if j >= spec[i+1][1]]
        seq_viol[viol_ids_i] = np.random.choice(
            [x+offset for x in xrange(0, 12) if x not in spec[-1][0]],
            len(viol_ids_i))

    return seq_viol


def generateNot2(seq, spec, offset, ratio=0.5, as_proll=False):
    if as_proll:
        ts_nt = np.argwhere(seq == 1)
        viol_ts_nt = ts_nt[np.random.choice(np.arange(len(ts_nt)),
                                            int(np.sum(seq > 0) * ratio),
                                            replace=False)]
        seq_viol = np.copy(seq)

        i = -1
        for i in xrange(len(spec)-1):
            # can be optimized by keeping track of indices visited...
            viol_ids_in = [j[0] for j in viol_ts_nt
                           if j[0] >= spec[i][1] and j[0] < spec[i+1][1] and
                           (j[1] % 12) in spec[i][0]]

            viol_ids_out = [j[0] for j in viol_ts_nt
                            if j[0] >= spec[i][1] and j[0] < spec[i+1][1] and
                            (j[1] % 12) not in spec[i][0]]

            seq_viol[viol_ids_in, np.random.choice(
                [x+offset for x in xrange(0, 12) if x not in spec[i][0]],
                len(viol_ids_in))] = 1

            seq_viol[viol_ids_out, np.random.choice(
                spec[i][0], len(viol_ids_out)) + offset] = 1

        viol_ids_in = [j[0] for j in viol_ts_nt
                       if j[0] >= spec[-1][1] and (j[1] % 12) in spec[i][0]]

        viol_ids_out = [j[0] for j in viol_ts_nt
                        if j[0] >= spec[i][1] and (j[1] % 12) not in spec[i][0]]

        seq_viol[viol_ids_in, np.random.choice(
            [x+offset for x in xrange(0, 12) if x not in spec[-1][0]],
            len(viol_ids_in))] = 1

        seq_viol[viol_ids_out, np.random.choice(
            spec[-1][0], len(viol_ids_out)) + offset] = 1

        # set modified notes to zero
        seq_viol[viol_ts_nt[:, 0], viol_ts_nt[:, 1]] = 0
    else:
        note_ids = np.where(seq >= 0)[0]
        viol_ids = np.random.choice(note_ids, int(np.sum(seq > 0) * ratio),
                                    replace=False)
        seq_viol = np.copy(seq)
        i = -1

        for i in xrange(len(spec)-1):
            viol_ids_in = [j for j in viol_ids
                           if (j >= spec[i][1] and
                               j < spec[i+1][1] and
                               (seq[j] % 12) in spec[i][0])]

            viol_ids_out = [j for j in viol_ids
                            if (j >= spec[i][1] and
                                j < spec[i+1][1] and
                                (seq[j] % 12) not in spec[i][0])]

            seq_viol[viol_ids_in] = np.random.choice(
                [x+offset for x in xrange(0, 12) if x not in spec[i][0]],
                len(viol_ids_in))

            seq_viol[viol_ids_out] = np.random.choice(
                spec[i][0], len(viol_ids_out)) + offset

        viol_ids_in = [j for j in viol_ids
                       if (j >= spec[-1][1] and
                           (seq[j] % 12) in spec[-1][0])]

        viol_ids_out = [j for j in viol_ids
                        if (j >= spec[i][1] and
                            (seq[j] % 12) not in spec[-1][0])]

        seq_viol[viol_ids_in] = np.random.choice(
            [x+offset for x in xrange(0, 12) if x not in spec[-1][0]],
            len(viol_ids_in))
        seq_viol[viol_ids_out] = np.random.choice(
            spec[-1][0], len(viol_ids_out)) + offset

    return seq_viol


def generate1(spec, n_pitches, n_timesteps, offset, as_proll=False):
    """Generates melodies according to experiment 1

    Parameters
    ----------
    spec : list of tuples ((pitch classes), int)
        List with pitch sets and their onset location in timesteps
    n_timesteps : int
        Number of timesteps in the melody

    Returns
    -------
    seq : np.ndarray
        piano roll containing the melody
    """
    if as_proll:
        seq = np.zeros((n_timesteps, 128), dtype=int)
        notes_ts = np.random.choice(
            np.arange(n_timesteps), n_pitches, replace=False)

        i = -1
        for i in xrange(len(spec)-1):
            cur_ts = [ts for ts in notes_ts
                      if ts >= spec[i][1] and ts < spec[i+1][1]]
            notes = np.random.choice(spec[i][0], np.sum(len(cur_ts))) + offset
            seq[cur_ts, notes] = 1

        cur_ts = [ts for ts in notes_ts if ts >= spec[-1][1]]
        notes = np.random.choice(spec[-1][0], np.sum(len(cur_ts))) + offset
        seq[cur_ts, notes] = 1
    else:
        # create a sequence of n_timesteps where 0 represents note and -1 rests
        seq = np.zeros(n_timesteps)-1
        seq[np.random.choice(np.arange(n_timesteps),
                             n_pitches, replace=False)] = 1
        i = -1
        for i in xrange(len(spec)-1):
            ids = seq[spec[i][1]: spec[i+1][1]] == 1
            seq[spec[i][1]: spec[i+1][1]][ids] = np.random.choice(spec[i][0],
                                                                  np.sum(ids))
        ids = seq[spec[i+1][1]:] == 1
        seq[spec[i+1][1]:][ids] = np.random.choice(spec[-1][0], np.sum(ids))
        seq[seq >= 0] += offset

    return seq


def generateSequence(intervals, min_len, max_len, note_resolution=12, clip=False):
    """Generates all training data based on intervals
    
    PARAMETERS
    ----------
    intervals : list
        List of intervals to use to create the sequence
    min_len : int
        Minimum length of sequence
    max_len : int
        Maximum length of sequence
    note_resolution : int
        Number of output "classes". Sequences will be modulo this number

    RETURNS
    -------
    input_data : numpy array
        List of sequences given specifications less last note
    target_data : numpy array
        Batch of last notes of sequences given specifications
    masks : numpy array
        Bach of masks used to set sequence lengths
    """

    # build sequence given intervals
    scale = np.zeros((len(intervals), note_resolution))
    scale[np.arange(len(intervals)), np.cumsum(intervals) % note_resolution] = 1
    # shift to obtain different modes
    # seqs = [np.roll(scale, -i, axis=0) for i in xrange(len(scale))]
    seqs = [scale]
    # shift to obtain different tonics
    for i in xrange(len(seqs)):
        for j in xrange(1, note_resolution):
            sequence = np.roll(seqs[i], -j, axis=1)
            seqs.append(sequence)
    # add reverse of all scales
    for i in xrange(len(seqs)):
        seqs.append(seqs[i][::-1])
    # create all possible masks per sequence of length [min_len, max_len]
    masks = []
    for i in xrange(min_len, max_len):
        mask = np.zeros((len(intervals), ), dtype=np.int32)
        mask[:i] = 1
        masks.append(mask)
        # not necessary to shift masks because sequences are shifted
        # for j in xrange(0, 1+max_len - i):
        #    masks.append(np.roll(mask, j))

    # cartesian product between masks and sequences, adding target 
    input_data, target_data, mask_data = [], [], []
    for mask in masks:
        for seq in seqs:
            if clip:
                temp = np.copy(seq)
                temp[np.count_nonzero(mask):] = 0.0
                input_data.append(temp)
            else:
                input_data.append(seq)
            target_data.append(seq[np.count_nonzero(mask)])
            mask_data.append(mask)

    input_data = np.array(input_data)
    target_data = np.array(target_data, dtype=np.int32)
    mask_data = np.array(mask_data, dtype=np.int32)

    return input_data, target_data, mask_data
        

def generateSequenceIter(batch_size, intervals, min_len, max_len,
                     note_resolution=12):
    """Generates training data based on intervals.

    PARAMETERS
    ----------
    batch_size : int
        Number of sequences to generate
    intervals : list
        List of intervals to use to create the sequence
    note_resolution : int
        Number of output "classes". Sequences will be modulo this number

    RETURNS
    -------
    input_data : iterator
        Batch of sequences given specifications less last note
    target_data : iterator
        Batch of last notes of sequences given specifications
    masks : masks
        Bach of masks used to set sequence lengths
    """

    # build sequence given intervals
    scale = np.zeros((len(intervals), note_resolution))
    scale[np.arange(len(intervals)), np.cumsum(intervals) % note_resolution] = 1

    while True:
        # input_data = []
        # target_data = []
        seq = np.zeros((batch_size, len(intervals), note_resolution),
                       dtype=np.int32)
        masks = np.zeros((batch_size, len(intervals)), dtype=np.int32)
        rnd_tonic = np.random.randint(0, len(intervals), batch_size)
        rnd_mode = np.random.randint(0, len(intervals), batch_size)
        rnd_lens = np.random.randint(min_len, max_len, batch_size)
        rnd_invs = 2 * np.random.randint(0, 2, batch_size) - 1
        targets = np.zeros((batch_size, note_resolution), dtype=np.int32)
        for i in xrange(batch_size):
            # shift to obtain different mode
            sequence = np.roll(scale, -rnd_mode[i], axis=0)
            # shift to obtain different tonic
            sequence = np.roll(scale, -rnd_tonic[i], axis=1)
            # randomize length
            # sequence = sequence[:rnd_lens[i]]
            # randomize reverse
            sequence = sequence[::rnd_invs[i]]
            # input_data.append(sequence[:-1])
            # target_data.append(sequence[-1])
            seq[i] = sequence
            masks[i, :rnd_lens[i]] = 1
            targets[i] = seq[i, rnd_lens[i]]
        yield seq, targets, masks


def generate1RNN(spec, n_pitches, n_timesteps, offset, as_proll=False,
                 note_resolution=128):
    """Generates melodies according to experiment 1

    PARAMETERS
    ----------
    spec : list of tuples ((pitch classes), int)
        List with pitch sets and their onset location in timesteps
    n_pitches : int
        Desired number of pitches in the generated melody
    n_timesteps : int
        Desired number of timesteps in the melody
    offset : int
        Offset for generated melody

    RETURNS
    -------
    seq : np.ndarray
        piano roll containing the melody
    """
    if as_proll:
        # melody buffer
        seq = np.zeros((n_timesteps, note_resolution), dtype=int)
        # choose time steps that will have note events
        notes_ts = np.random.choice(
            np.arange(n_timesteps), n_pitches, replace=False)

        for i in xrange(len(spec)-1):
            cur_ts = [ts for ts in notes_ts
                      if ts >= spec[i][1] and ts < spec[i+1][1]]
            notes = np.random.choice(spec[i][0], np.sum(len(cur_ts))) + offset
            seq[cur_ts, notes] = 1

        cur_ts = [ts for ts in notes_ts if ts >= spec[-1][1]]
        notes = np.random.choice(spec[-1][0], np.sum(len(cur_ts))) + offset
        seq[cur_ts, notes] = 1
        error = np.zeros((n_timesteps, 128), dtype=int)
        error[np.random.choice([x+offset for x in xrange(12)
                                if x not in spec[-1][0]])] = 1
        return seq, error
    else:
        # create a sequence of n_timesteps where 0 represents note and -1 rests
        seq = np.zeros(n_timesteps)-1
        seq[np.random.choice(np.arange(n_timesteps),
                             n_pitches, replace=False)] = 1
        i = -1
        for i in xrange(len(spec)-1):
            ids = seq[spec[i][1]: spec[i+1][1]] == 1
            seq[spec[i][1]: spec[i+1][1]][ids] = np.random.choice(spec[i][0],
                                                                  np.sum(ids))
        ids = seq[spec[i+1][1]:] == 1
        seq[spec[i+1][1]:][ids] = np.random.choice(spec[-1][0], np.sum(ids))
        seq[seq >= 0] += offset
        error = np.random.choice([x for x in xrange(12) if x not in
                                  spec[-1][0]])
        return seq, error + offset


def generate2(spec, n_timesteps, offset, start_harm=True, as_proll=False):
    """Generates melodies according to experiment 2

    Parameters
    ----------
    spec : list of tuples ((pitch classes), int)
        List with pitch sets and their onset location in timesteps
    n_timesteps : int
        Number of timesteps in the melody

    Returns
    -------
    seq : np.ndarray
        piano roll containing the melody
    """
    if as_proll:
        seq = np.zeros((n_timesteps, 128), dtype=int)

        i = -1
        for i in xrange(len(spec)-1):
            if start_harm:
                in_ids = np.arange(spec[i][1], spec[i+1][1], 2)
                out_ids = np.arange(spec[i][1]+1, spec[i+1][1], 2)
            else:
                out_ids = np.arange(spec[i][1], spec[i+1][1], 2)
                in_ids = np.arange(spec[i][1]+1, spec[i+1][1], 2)

            out_tones = [x for x in xrange(0, 12) if x not in spec[i][0]]

            notes = np.hstack((np.random.choice(spec[i][0], len(in_ids)),
                               np.random.choice(out_tones, len(out_ids))))

            seq[np.arange(spec[i][1], spec[i+1][1]), notes+offset] = 1

        if start_harm:
            in_ids = np.arange(spec[i][1], spec[i+1][1], 2)
            out_ids = np.arange(spec[i][1]+1, spec[i+1][1], 2)
        else:
            out_ids = np.arange(spec[i][1], spec[i+1][1], 2)
            in_ids = np.arange(spec[i][1]+1, spec[i+1][1], 2)

        out_tones = [x for x in xrange(0, 12) if x not in spec[-1][0]]
        notes = np.hstack((np.random.choice(spec[-1][0], len(in_ids)),
                           np.random.choice(out_tones, len(out_ids))))
        seq[np.arange(spec[-1][1], n_timesteps), notes+offset] = 1
    else:
        # create a sequence of n_timesteps where 0 represents note and -1 rests
        seq = np.zeros(n_timesteps)

        i = -1
        for i in xrange(len(spec)-1):
            if start_harm:
                in_ids = np.arange(spec[i][1], spec[i+1][1], 2)
                out_ids = np.arange(spec[i][1]+1, spec[i+1][1], 2)
            else:
                out_ids = np.arange(spec[i][1], spec[i+1][1], 2)
                in_ids = np.arange(spec[i][1]+1, spec[i+1][1], 2)

            out_tones = [x for x in xrange(0, 12) if x not in spec[i][0]]
            seq[in_ids] = np.random.choice(spec[i][0], len(in_ids))
            seq[out_ids] = np.random.choice(out_tones, len(out_ids))

        if start_harm:
            in_ids = np.arange(spec[-1][1], n_timesteps, 2)
            out_ids = np.arange(spec[-1][1]+1, n_timesteps, 2)
        else:
            in_ids = np.arange(spec[-1][1]+1, n_timesteps, 2)
            out_ids = np.arange(spec[-1][1], n_timesteps, 2)

        out_tones = [x for x in xrange(0, 12) if x not in spec[-1][0]]
        seq[in_ids] = np.random.choice(spec[-1][0], len(in_ids))
        seq[out_ids] = np.random.choice(out_tones, len(out_ids))
        seq[seq >= 0] += offset
    return seq


def generateData(experiment, specs, n_pitches, n_timesteps, offset, n_obs,
                 as_proll=False):
    if experiment == 1:
        seq_dict = {}
        seq_dict['with_specs'] = np.array([generate1(
            specs, n_pitches, n_timesteps, offset, as_proll=as_proll)
            for x in xrange(n_obs)],
            dtype='int32')
        seq_dict['without_specs'] = np.array([generateNot1(
            seq, specs, offset, as_proll=as_proll)
            for seq in seq_dict['with_specs']],
            dtype='int32')
    elif experiment == 2:
        seq_dict = {}
        seq_dict['with_specs'] = np.array([generate2(
            specs, n_timesteps, offset, as_proll=as_proll)
            for x in xrange(n_obs)],
            dtype='int32')
        seq_dict['without_specs'] = np.array([generateNot2(
            seq, specs, offset, as_proll=as_proll)
            for seq in seq_dict['with_specs']],
            dtype='int32')
    return seq_dict


def generateMask(data, max_length):
    mask = np.zeros((len(data), max_length), dtype='int32')
    for i in xrange(len(data)):
        mask[i][np.arange(len(data[i]))] = 1
    return mask


def generateDataRNN(experiment, specs, n_pitches, n_timesteps, offset, n_obs,
                    min_len, max_len, as_proll=False):
    if experiment == 1:
        seq_dict = {}
        # generate melodies with specs
        seq_dict['with_specs'] = [generate1(
            specs, n_pitches, n_timesteps, offset, as_proll=as_proll)
            for _ in xrange(n_obs)]

        # modify melodies not to satisfy specs
        seq_dict['without_specs'] = [generateNot1(
            seq, specs, offset, as_proll=as_proll)
            for seq in seq_dict['with_specs']]

        # generate masks for the melodies
        lens = np.random.randint(min_len, max_len, n_obs)
        seq_dict['masks'] = np.zeros((n_obs, max_len), dtype=int)

        # create mask and save only last note from with_specs
        # the plan is that the rnn will learn to modify notes such that they
        # satisfy the specifications
        for i in xrange(len(seq_dict['with_specs'])):
            seq_dict['with_specs'][i] = seq_dict['with_specs'][i][lens[i]-1]
            seq_dict['masks'][i][:lens[i]] = 1

        seq_dict['without_specs'] = np.array(seq_dict['without_specs']).reshape(
            (n_obs, n_timesteps,  len(seq_dict['without_specs'][0].shape)))
        seq_dict['with_specs'] = np.array(seq_dict['with_specs'])
    elif experiment == 2:
        seq_dict = {}
        seq_dict['with_specs'] = np.array([generate2(
            specs, n_timesteps, offset, as_proll=False)
            for x in xrange(n_obs)],
            dtype='int32')
        seq_dict['without_specs'] = np.array([generateNot2(
            seq, specs, offset, as_proll=False)
            for seq in seq_dict['with_specs']],
            dtype='int32')
    return seq_dict
