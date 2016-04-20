from __future__ import division
import numpy as np


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


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


def generate1RNN(spec, n_pitches, n_timesteps, offset, as_proll=False):
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
            specs, n_pitches, n_timesteps, offset, as_proll=as_proll).astype(
            'float32') for _ in xrange(n_obs)]

        # modify melodies not to satisfy specs
        seq_dict['without_specs'] = [generateNot1(
            seq, specs, offset, as_proll=as_proll).astype('float32')
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

        seq_dict['without_specs'] = np.array(seq_dict['without_specs'],
        dtype='float32').reshape(
            (n_obs, n_timesteps,  len(seq_dict['without_specs'][0].shape)))
        seq_dict['with_specs'] = np.array(seq_dict['with_specs'],
        dtype='float32')
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
