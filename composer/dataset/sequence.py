'''
High-level data-structures for representing MIDI-like sequences of notes 
along with their associated event encodings.

'''

from enum import Enum
from pathlib import Path
from pretty_midi import PrettyMIDI
from composer.exceptions import InvalidParameterError

class Note:
    '''
    A representation of a musical note on a sequence.

    '''

    def __init__(self, start, end, pitch, velocity):
        '''
        Initializes an instance of :class:`Note`.

        :param start:
            The start time of the note, in milliseconds.
        :param end:
            The end time of the note, in milliseconds.
        :param pitch:
            The pitch of the note (as a MIDI pitch: from 0 to 127).
        :param velocity:
            The velocity of the note (as a MIDI velocity: from 0 to 127).
        '''

        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity

    def __repr__(self):
        return 'Note(start={:f}, end={:f}, pitch={}, velocity={})'.format(
            self.start, self.end, self.pitch, self.velocity)

    @property
    def duration(self):
        '''
        The duration of the note, in milliseconds.

        '''

        return self.end - self.start

class EventType(Enum):
        '''
        The type of an :class:`Event`.
        
        '''

        NOTE_ON = 0
        NOTE_OFF = 1,
        TIME_SHIFT = 2
        VELOCITY = 3

class Event:
    '''
    An event representing some change on the NoteSequence.
    Every event has a type and value associated with it.

    '''

    def __init__(self, event_type, value):
        '''
        Initialize an instance of :class:`Event`.

        :param event_type:
            An :class:`EventType` value representing the type of the event.
        :param value:
            The value of the event. The type of this object depends on the 
            type of the event.

        '''
        
        self.type = event_type
        self.value = value

    def __repr__(self):
        return 'Event(type={}, value={})'.format(self.type, self.value)

class NoteSequence:
    '''
    A MIDI-like sequence representation.

    '''

    def __init__(self, notes=None):
        '''
        Initializes an instance of :class:`NoteSequence`.

        :param notes:
            A list of :class:`Note` objects to add to the sequence.
        '''

        self.notes = []
        if notes is not None:
            self.add_notes(notes, False)
        
            # We manually reorder the notes since we disabled resorting in the add_notes call.
            # This is an optimization so that we only have to sort once: at the end.
            self.notes.sort(key=lambda x: x.start)

    def add_notes(self, notes, maintain_order=True):
        '''
        Adds a list of notes to this :class:`NoteSequence`.

        :param notes:
            A list of :class:`Note` objects to add to this :class:`NoteSequence`.
        :param maintain_order:
            Indicates whether the internal representation of notes should be resorted so that
            it is in ascending order with respect to the starting time of each note. Defaults
            to ``True``.
        '''

        self.notes.extend(notes)
        if not maintain_order: return
        self.notes.sort(key=lambda x: x.start)

    def to_event_sequence(self, time_step_increment=10, max_time_steps=100, velocity_bins=32):
        '''
        Computes the event-based representation of this :class:`NoteSequence`. 

        :param time_step_increment:
            The number of milliseconds that a single step in time represents.
            Defaults to 10 milliseconds (i.e. one step in time is 10 milliseconds).
        :param max_time_steps:
            The maximum number of time steps that a single event can shift time by.
            Defaults to 100 (i.e. time shift can vary from 1 to 100 time steps).
            If this is ``None``, there is no limit.

            This exists because when encoding events as one-hot vectors for use in
            the deep learning models, we must have a finite range of values for which
            the value of the time shift event can take on.
        :param velocity_bins:
            The number of bins to quantize the note velocity values into. Defaults to 32.
        :returns:
            A list of :class:`Event` objects.

        '''

        events = []

        # Make sure that the notes are in sorted order.
        # If they aren't then our events will also be in the wrong order!
        ordered_notes = sorted(self.notes, key=lambda x: x.start)

        # We need to split each note in two separate events: on and off.
        # This way, when we sort based on time, we don't run into issues
        # where notes overlap and therefore we would have to step BACK in time.
        # 
        # Since time shifts cannot be negative, we need to organize our events
        # in order of time (going forward). For example, suppose we have two notes:
        # one from 0 to 1 second and another from 0 to 5 seconds. Our event sequence
        # should be:
        #   [NOTE_ON(1), NOTE_ON(2), TIME_SHIFT(1), NOTE_OFF(1), TIME_SHIFT(4), NOTE_OFF(2)].
        #
        # Notice how there are TWO time shift events since we need to turn off the first note
        # before we shift to the end of the second note and turn it off as well. On the other hand,
        # if we were to just sort based on the starting time of the notes and then add time shifts
        # between the start and end of each note, we would get an (incorrect) event sequence such as:
        #   [NOTE_ON(1), TIME_SHIFT(1), NOTE_OFF(1), NOTE_ON(2), TIME_SHIFT(5), NOTE_OFF(2)].
        #
        # Of course, this "local" approach to building the event sequence is flawed since it fails to
        # consider the CURRENT TIME. If it did, it would become immediately obvious that the NOTE_ON(2)
        # event occurs at 1 second and that NOTE_OFF(2) occurs at 6 seconds. 
        
        # We split each note into two separate events: NOTE_ON and NOTE_OFF; however, rather than
        # using the Event class directly, we use a tuple structure: (EventType, float, Note).
        # This is because, we have yet to create the rest of the note events and thus, we still 
        # need the note data (such as velocity and pitch).
        note_events = []
        for note in ordered_notes:
            on = (EventType.NOTE_ON, note.start, note)
            off = (EventType.NOTE_OFF, note.end, note)
            note_events.extend([on, off])

        # Sort note events by time
        note_events.sort(key=lambda x: x[1])

        current_time = 0
        current_velocity = 0
        for note_event in note_events:
            note_event_type, note_event_time, note = note_event         
            # The interval of time between current time and the event, in time steps.
            interval = int(round(note_event_time - current_time) / time_step_increment)

            # If the interval of time exceeds the max time steps, we need to break it up into
            # multiple time shift events...
            for i in range(interval // max_time_steps):
                events.append(Event(EventType.TIME_SHIFT, max_time_steps))

            # Get the remaining time and if it isn't zero, add the time shift command.
            interval %= max_time_steps
            if interval > 0:
                events.append(Event(EventType.TIME_SHIFT, interval))

            # If the note velocity is different from our current velocity,
            # we add a velocity event to indicate a change in velocity.
            if current_velocity != note.velocity:
                # Scale the velocity value so that it is in a velocity bin.
                # There are 128 possible velocity values: 0 to 127.
                velocity_bin = (note.velocity * velocity_bins) // 128
                events.append(Event(EventType.VELOCITY, velocity_bin))

            
            # Add note ON/OFF event
            events.append(Event(note_event_type, note.pitch))

            current_time = note_event_time
            current_velocity = note.velocity

        return events

    @staticmethod
    def from_midi(filepath, programs=None, ignore_drums=True):
        '''
        Creates a :class:`NoteSequence` from a MIDI file.

        :param filepath:
            The path of the MIDI file to load.
        :param programs:
            A list of integers representing the MIDI programs to load notes from.
            By default, this is `None` which means that ALL programs in the MIDI file will be used. 
        :param ignore_drums:
            Indicates whether drums should be excluded. Defaults to True.
        :returns:
            An instance of :class:`NoteSequence` representing the MIDI file.
        '''

        filepath = Path(filepath)
        if not filepath.is_file():
            raise InvalidParameterError('Cannot create NoteSequence from \'{}\' since it is not a file.'.format(filepath))
    
        with open(filepath, 'rb') as file:
            midi = PrettyMIDI(file)
            notes = []
            for instrument in midi.instruments:
                if ignore_drums and instrument.is_drum: continue
                if programs is not None and not instrument.program in programs: continue

                for midi_note in instrument.notes:
                    # PrettyMIDI timing is in seconds so we have to convert them to milliseconds.
                    notes.append(Note(midi_note.start * 1000, midi_note.end * 1000, midi_note.pitch, midi_note.velocity))
                
            return NoteSequence(notes)