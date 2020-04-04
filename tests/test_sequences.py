'''
Tests the ``composer.datset.sequence`` module.

'''

import numpy as np
import composer.dataset.sequence as sequence


def _compare_note_sequences(notes_a, notes_b):
    '''
    Compares two :class:`composer.dataset.sequence.NoteSequence` objects.

    :returns:
        ``True`` if they are equal (based on their attributes); ``False`` otherwise.
        
    '''
    
    if len(notes_a.notes) != len(notes_b.notes): return False
    for i in range(len(notes_a.notes)):
        note_a, note_b = notes_a.notes[i], notes_b.notes[i]
        if note_a.start != note_b.start: return False
        if note_a.end != note_b.end: return False
        if note_a.pitch != note_b.pitch: return False
        if note_a.velocity != note_b.velocity: return False

    if len(notes_a.sustain_periods) != len(notes_b.sustain_periods): return False
    for i in range(len(notes_a.sustain_periods)):
        period_a, period_b = notes_a.sustain_periods[i], notes_b.sustain_periods[i]
        if period_a.start != period_b.start: return False
        if period_a.end != period_b.end: return False

    return True

def test_note_sequence_time_stretch():
    stretch_factors = [0.50, 1.0, 1.5]
    for stretch_factor in stretch_factors:
        note_sequence = sequence.NoteSequence([
            sequence.Note(0, 2000, 2, 0),
            sequence.Note(3000, 4000, 1, 0)
        ])

        target_note_sequence = sequence.NoteSequence([
            sequence.Note(0, 2000 * stretch_factor, 2, 0),
            sequence.Note(3000 * stretch_factor, 4000 * stretch_factor, 1, 0)
        ])

        # Test regular operation
        modified_note_sequence = note_sequence.time_stretch(stretch_factor, inplace=False)
        assert _compare_note_sequences(modified_note_sequence, target_note_sequence)

        # Test inplace operation
        note_sequence.time_stretch(stretch_factor, inplace=True)
        assert _compare_note_sequences(note_sequence, target_note_sequence)

def test_note_sequence_pitch_shift():
    offsets = [0, 3, 1000, -2]
    for offset in offsets:
        note_sequence = sequence.NoteSequence([
            sequence.Note(0, 2000, 2, 0),
            sequence.Note(3000, 4000, 1, 0)
        ])

        target_note_sequence = sequence.NoteSequence([
            sequence.Note(0, 2000, np.clip(2 + offset, 0, 127), 0),
            sequence.Note(3000, 4000, np.clip(1 + offset, 0, 127), 0)
        ])

        # Test regular operation
        modified_note_sequence = note_sequence.pitch_shift(offset, inplace=False)
        assert _compare_note_sequences(modified_note_sequence, target_note_sequence)
        # Test inplace operation
        
        note_sequence.pitch_shift(offset, inplace=True)
        assert _compare_note_sequences(note_sequence, target_note_sequence)
