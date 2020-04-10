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

def _compare_event_sequences(events_a, events_b):
    '''
    Compares two :class:`composer.dataset.sequence.EventSequence` objects.

    :returns:
        ``True`` if they are equal (based on their attributes); ``False`` otherwise.
        
    '''
    
    if events_a.time_step_increment != events_b.time_step_increment: return False
    if events_a.max_time_steps != events_b.max_time_steps: return False
    if events_a.velocity_bins != events_b.velocity_bins: return False

    if len(events_a.events) != len(events_b.events): return False
    for i in range(len(events_b.events)):
        event_a, event_b = events_a.events[i], events_b.events[i]
        if event_a.type != event_b.type: return False
        if event_a.value != event_b.value: return False

    return True

# Constants for conversion.
# These don't really need to be varied in testing since they act
# almost as a scale on the time shift and velocity events respectively.
_TIME_STEP_INCREMENT = 10
_MAX_TIME_STEPS = 100
# Velocity is split into 4 bins: [0, 31], [32, 63], [64, 95], [96-127]
_VELOCITY_BINS = 4

def test_note_sequence_to_event_sequence():
    # Test with notes but no sustain periods
    note_sequence_a = sequence.NoteSequence([
        sequence.Note(0, 2000, 2, 64), # Velocity bin index is 2
        sequence.Note(3000, 4000, 1, 9) # Velocity bin index is 0
    ])

    target_event_sequence_a = sequence.EventSequence([
        # Turn on note with pitch 2 for 2 seconds (200 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 2),
        sequence.Event(sequence.EventType.NOTE_ON, 2),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 2),
        # Wait for 1 second (100 time steps) before turning on next note.
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        # Turn on note with pitch 1 for 1 second (100 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 0),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 1)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    event_sequence_a = note_sequence_a.to_event_sequence(_TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)
    assert _compare_event_sequences(event_sequence_a, target_event_sequence_a)

    # Test with notes and sustain periods
    note_sequence_b = sequence.NoteSequence([
        sequence.Note(0, 4000, 1, 37), # Velocity bin index is 1
        sequence.Note(0, 4000, 4, 37), # Velocity bin index is 1
        sequence.Note(5000, 11000, 3, 96) # Velocity bin index is 3
    ], [
        sequence.SustainPeriod(4000, 5000)
    ])

    target_event_sequence_b = sequence.EventSequence([
        # Turn on note with pitch 1 for 4 seconds (400 time steps).
        # Turn on note with pitch 4 for 4 seconds (400 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        # Sustain period for 1 second (100 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.NOTE_OFF, 1),
        sequence.Event(sequence.EventType.NOTE_OFF, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Turn on note with pitch 3 for 6 second (600 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 3),
        sequence.Event(sequence.EventType.NOTE_ON, 3),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 3)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    event_sequence_b = note_sequence_b.to_event_sequence(_TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)
    assert _compare_event_sequences(event_sequence_b, target_event_sequence_b)

    # Test with no notes but with sustain periods
    note_sequence_c = sequence.NoteSequence(None, [
        sequence.SustainPeriod(0, 1000),
        sequence.SustainPeriod(2500, 5670),
        sequence.SustainPeriod(8000, 10000),
    ])

    target_event_sequence_c = sequence.EventSequence([
        # Sustain period for 1 second (100 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Wait for 1.5 seconds (150 time steps).
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 50),
        # Sustain period for 31.7 seconds (317 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 17),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Wait for 2.33 seconds (233 time steps).
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 33),
        # Sustain period for 2 seconds (200 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    event_sequence_c = note_sequence_c.to_event_sequence(_TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)
    assert _compare_event_sequences(event_sequence_c, target_event_sequence_c)

def test_event_sequence_to_note_sequence():
    # Test with notes but no sustain periods
    event_sequence_a = sequence.EventSequence([
        # Turn on note with pitch 2 for 2 seconds (200 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 2),
        sequence.Event(sequence.EventType.NOTE_ON, 2),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 2),
        # Wait for 1 second (100 time steps) before turning on next note.
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        # Turn on note with pitch 1 for 1 second (100 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 0),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 1)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    target_note_sequence_a = sequence.NoteSequence([
        sequence.Note(0, 2000, 2, 64), # Velocity bin index is 2
        sequence.Note(3000, 4000, 1, 0) # Velocity bin index is 0
    ])

    note_sequence_a = event_sequence_a.to_note_sequence()
    assert _compare_note_sequences(note_sequence_a, target_note_sequence_a)

    # Test with notes and sustain periods
    event_sequence_b = sequence.EventSequence([
        # Turn on note with pitch 1 for 4 seconds (400 time steps).
        # Turn on note with pitch 4 for 4 seconds (400 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        # Sustain period for 1 second (100 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.NOTE_OFF, 1),
        sequence.Event(sequence.EventType.NOTE_OFF, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Turn on note with pitch 3 for 6 second (600 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 3),
        sequence.Event(sequence.EventType.NOTE_ON, 3),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 3)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    target_note_sequence_b = sequence.NoteSequence([
        sequence.Note(0, 4000, 1, 32), # Velocity bin index is 1
        sequence.Note(0, 4000, 4, 32), # Velocity bin index is 1
        sequence.Note(5000, 11000, 3, 96) # Velocity bin index is 3
    ], [
        sequence.SustainPeriod(4000, 5000)
    ])

    note_sequence_b = event_sequence_b.to_note_sequence()
    assert _compare_note_sequences(note_sequence_b, target_note_sequence_b)

    # Test with no notes but with sustain periods
    event_sequence_c = sequence.EventSequence([
        # Sustain period for 1 second (100 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Wait for 1.5 seconds (150 time steps).
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 50),
        # Sustain period for 31.7 seconds (317 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 17),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Wait for 2.33 seconds (233 time steps).
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 33),
        # Sustain period for 2 seconds (200 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    target_note_sequence_c = sequence.NoteSequence(None, [
        sequence.SustainPeriod(0, 1000),
        sequence.SustainPeriod(2500, 5670),
        sequence.SustainPeriod(8000, 10000),
    ])

    note_sequence_c = event_sequence_c.to_note_sequence()
    assert _compare_note_sequences(note_sequence_c, target_note_sequence_c)

def test_event_to_id():
    event_sequence = sequence.EventSequence([
        # Turn on note with pitch 1 for 4 seconds (400 time steps).
        # Turn on note with pitch 4 for 4 seconds (400 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        # Sustain period for 1 second (100 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.NOTE_OFF, 1),
        sequence.Event(sequence.EventType.NOTE_OFF, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Turn on note with pitch 3 for 6 second (600 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 3),
        sequence.Event(sequence.EventType.NOTE_ON, 3),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 3)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    target_event_ids = [
        257, 1, 4, 359, 359, 359, 359, 360, 129, 132, 359,
        361, 259, 3, 359, 359, 359, 359, 359, 359, 131
    ]

    event_ids = []
    for event in event_sequence.events:
        event_ids.append(sequence.IntegerEncodedEventSequence.event_to_id(
            event.type, event.value, event_sequence.event_ranges,
            event_sequence.event_value_ranges
        ))
    
    assert event_ids == target_event_ids

def test_id_to_event():
    target_event_sequence = sequence.EventSequence([
        # Turn on note with pitch 1 for 4 seconds (400 time steps).
        # Turn on note with pitch 4 for 4 seconds (400 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        # Sustain period for 1 second (100 time steps).
        sequence.Event(sequence.EventType.SUSTAIN_ON, None),
        sequence.Event(sequence.EventType.NOTE_OFF, 1),
        sequence.Event(sequence.EventType.NOTE_OFF, 4),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.SUSTAIN_OFF, None),
        # Turn on note with pitch 3 for 6 second (600 time steps).
        sequence.Event(sequence.EventType.VELOCITY, 3),
        sequence.Event(sequence.EventType.NOTE_ON, 3),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 3)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    event_ids = [
        257, 1, 4, 359, 359, 359, 359, 360, 129, 132, 359,
        361, 259, 3, 359, 359, 359, 359, 359, 359, 131
    ]

    events = []
    for event_id in event_ids:
        events.append(sequence.IntegerEncodedEventSequence.id_to_event(
            event_id, target_event_sequence.event_ranges,
            target_event_sequence.event_value_ranges
        ))

    event_sequence = sequence.EventSequence(events, _TIME_STEP_INCREMENT, 
                                            _MAX_TIME_STEPS, _VELOCITY_BINS)

    assert _compare_event_sequences(event_sequence, target_event_sequence)

def test_sustain_period_extension():
    note_sequence = sequence.NoteSequence([
        sequence.Note(0, 4000, 4, 64),
        sequence.Note(0, 4000, 1, 64),
        sequence.Note(0, 4000, 3, 64),
        sequence.Note(5000, 11000, 3, 64)
    ], [
        sequence.SustainPeriod(0, 6000)
    ])

    target_event_sequence = sequence.EventSequence([
        sequence.Event(sequence.EventType.VELOCITY, 2),
        sequence.Event(sequence.EventType.NOTE_ON, 4),
        sequence.Event(sequence.EventType.NOTE_ON, 1),
        sequence.Event(sequence.EventType.NOTE_ON, 3),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 4),
        sequence.Event(sequence.EventType.NOTE_OFF, 1),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.TIME_SHIFT, 100),
        sequence.Event(sequence.EventType.NOTE_OFF, 3)
    ], _TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS)

    event_sequence = note_sequence.to_event_sequence(_TIME_STEP_INCREMENT, _MAX_TIME_STEPS, _VELOCITY_BINS, 
                                                     sustain_period_encode_mode=sequence.NoteSequence.SustainPeriodEncodeMode.EXTEND)
    assert _compare_event_sequences(event_sequence, target_event_sequence)