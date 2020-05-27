'''
High-level data-structures for representing MIDI-like sequences of notes 
along with their associated event encodings.

'''

import os
import abc
import copy
import array
import struct
import itertools
import collections
import numpy as np

from pathlib import Path
from enum import Enum, IntEnum, unique
from pretty_midi import PrettyMIDI, Instrument, Note as MIDINote, ControlChange
from composer.exceptions import InvalidParameterError

class Note:
    '''
    A representation of a musical note on a sequence.

    :ivar start:
        The start time of the note, in milliseconds.
    :ivar end:
        The end time of the note, in milliseconds.
    :ivar pitch:
        The pitch of the note (as a MIDI pitch: from 0 to 127).
    :ivar velocity:
        The velocity of the note (as a MIDI velocity: from 0 to 127).

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

class EventType(IntEnum):
        '''
        The type of an :class:`Event`.
        
        :cvar NOTE_ON:
            Indicates a note is active (enabled).
        :cvar NOTE_OFF:
            Indicates a note is inactive (disabled).
        :cvar TIME_SHIFT:
            Indicates a shift in time (in time steps).
        :cvar VELOCITY:
            Indicates a change in velocity. 
        :cvar SUSTAIN_ON:
            Indicates the start of a sustain period.
        :cvar SUSTAIN_OFF:
            Indicates the end of a sustain period.
        
        '''

        NOTE_ON = 1
        NOTE_OFF = 2
        TIME_SHIFT = 3
        VELOCITY = 4
        SUSTAIN_ON = 5
        SUSTAIN_OFF = 6

        @staticmethod
        def make_int_type_map():
            '''
            Creates a ``dict`` that maps the integer values to their ``EventType`` objects.

            '''

            mapping = {}
            enum_values = list(EventType)
            for enum_value in enum_values:
                mapping[enum_value.value] = enum_value

            return mapping

# A handle dictionary that maps integers to their event type objects.
_EVENT_TYPE_MAPPINGS = EventType.make_int_type_map()

class Event:
    '''
    An event representing some change on the NoteSequence.
    Every event has a type and value associated with it.

    :cvar NONE_VALUE:
        A reserved value for representing `None` values as an integer.
    :ivar type:
        An :class:`EventType` value representing the type of the event.
    :ivar value:
        Data associated with the event (i.e. a parameter).

    '''

    NONE_VALUE = -1

    def __init__(self, event_type, value):
        '''
        Initialize an instance of :class:`Event`.

        :param event_type:
            An :class:`EventType` value representing the type of the event.
        :param value:
            The value of the event. The type of this object depends on the 
            type of the event.

            Note that the value of an event is not restricted. It can be any 
            object so long as the object can be represented as int. Also, 
            when the event is decoded (after being encoded), the all values
            will be in the form of integers. Therefore, if you are using
            custom objects as event values, it is your responsibility
            to convert these values back to their original objects.

            The event value system is primarily designed to store integer 
            values only. While it can be extended past this, it is not
            supported and therefore requires a lot of manual maintenance.

        '''
        
        self.type = event_type
        self.value = value

    @staticmethod
    def encode_value(event):
        '''
        Encodes the value of an event as an integer.

        '''

        value = Event.NONE_VALUE
        if event.value is not None:
            value = int(event.value)

        return value

    @staticmethod
    def decode_value(value):
        '''
        Decodes an event value.

        '''
        
        return value if value != Event.NONE_VALUE else None

    def __str__(self):
        return '{}<{}>'.format(self.type.name, self.value)

    def __repr__(self):
        return 'Event(type={}, value={})'.format(str(self.type), self.value)

class SustainPeriod:
    '''
    A period of time where the sustain pedal is active.

    :ivar start:
        The start time, in milliseconds, of the sustain period.
    :ivar end:
        The end time, in milliseconds, of the sustain period.

    '''

    def __init__(self, start, end):
        '''
        Initializes an instance of :class:`SustainPeriod`.

        :param start:
            The start time, in milliseconds, of the sustain period.
        :param end:
            The end time, in milliseconds, of the sustain period.

        '''
        self.start = start
        self.end = end

    def __repr__(self):
        return 'SustainPeriod(start={}, end={})'.format(self.start, self.end)

class NoteSequence:
    '''
    A MIDI-like sequence representation.

    :ivar notes:
        A list of :class:`Note` objects in the sequence.
    :ivar sustain_periods:
        A list of :class:`SustainPeriod` objects in the sequence.

    '''

    @unique
    class SustainPeriodEncodeMode(Enum):
        '''
        The mode for encoding sustain periods.

        :cvar NONE:
            Indicates that sustain periods should not be encoded at all.
            That is, they should be ignored when creating note sequences
            from MIDI files.
        :cvar EXTEND:
            Indicates that all notes within a period of time where the sustain 
            pedal is active should be extended so that they span the entirety 
            of the  sustain pedal period or to the start of the next note of 
            the same pitch, whichever happens first.
        :cvar EVENTS:
            Indicates that sustain periods should be encoded as :class:`EventType.SUSTAIN_ON`
            and :class:`EventType.SUSTAIN_OFF` events.

        '''

        NONE = 'none'
        EXTEND = 'extend'
        EVENTS = 'events'

    def __init__(self, notes=None, sustain_periods=None):
        '''
        Initializes an instance of :class:`NoteSequence`.

        :param notes:
            A list of :class:`Note` objects to add to the sequence.
        :param sustain_periods:
            A list of :class:`SustainPeriod` objects to add to the sequence.
        '''

        self.notes = []
        if notes is not None:
            self.add_notes(notes, False)
        
            # We manually reorder the notes since we disabled resorting in the add_notes call.
            # This is an optimization so that we only have to sort once: at the end.
            self.notes.sort(key=lambda x: x.start)

        self.sustain_periods = sustain_periods if sustain_periods is not None else []

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

    def time_stretch(self, percent, inplace=True):
        '''
        Stretches this :class:`NoteSequence` in time by a factor of the original time.

        :param percent:
            The percent (0 to 1) of the original time to stretch.
        :param inplace:
            Indicates whether the operation should be applied inplace. Defaults to ``True``.

            Note that non-inplace operations require more memory since a deepcopy has to be performed.
        :returns:
            An instance of :class:`NoteSequence` with the applied modifications.

        '''

        _copy_func = lambda x: x if inplace else copy.deepcopy(x)
        notes = _copy_func(self.notes)
        sustain_periods = _copy_func(self.sustain_periods)

        for note in notes:
            note.start *= percent
            note.end *= percent

        for sustain_period in sustain_periods:
            sustain_period.start *= percent
            sustain_period.end *= percent

        return self if inplace else NoteSequence(notes, sustain_periods)

    def time_shift(self, offset, inplace=True):
        '''
        Stretches the start and end of all notes in this :class:`NoteSequence` by a specified offset.

        :param offset:
            The amount to shift the start and end of each note by.
        :param inplace:
            Indicates whether the operation should be applied inplace. Defaults to ``True``.
            
            Note that non-inplace operations require more memory since a deepcopy has to be performed.
        :returns:
            An instance of :class:`NoteSequence` with the applied modifications.

        '''
        
        _copy_func = lambda x: x if inplace else copy.deepcopy(x)
        notes = _copy_func(self.notes)
        sustain_periods = _copy_func(self.sustain_periods)

        for note in notes:
            note.start += offset
            note.end += offset

        for sustain_period in sustain_periods:
            sustain_period.start += offset
            sustain_period.end += offset

        return self if inplace else NoteSequence(notes, sustain_periods)

    def trim_start(self, inplace=True):
        '''
        Trims silence from the beginning of the note sequence.
        
        :param inplace:
            Indicates whether the operation should be applied inplace. Defaults to `True`.

            Note that non-inplace operations require more memory since a deepcopy has to be performed.
         :returns:
            An instance of :class:`NoteSequence` with the applied modifications.   

        '''

        offset = self.notes[0].start
        if len(self.sustain_periods) > 0:
            offset = min(offset, self.sustain_periods[0].start)

        return self.time_shift(-offset, inplace=inplace)

    def pitch_shift(self, offset, inplace=True):
        '''
        Shifts the pitch of all notes in this :class:`NoteSequence` by a specified offset.

        :note:
            This will clamp the pitch of a note between 0 and 127; that is, if a note has a negative
            pitch or one that exceeds 127, it will have a final pitch of 0 or 127 respectively.

        :param offset:
            The amount to shift the pitch of each note by.
        :param inplace:
            Indicates whether the operation should be applied inplace. Defaults to ``True``.

            Note that non-inplace operations require more memory since a deepcopy has to be performed.
        :returns:
            An instance of :class:`NoteSequence` with the applied modifications.

        '''

        _copy_func = lambda x: x if inplace else copy.deepcopy(x)
        notes = _copy_func(self.notes)

        for note in notes:
            note.pitch = np.clip(note.pitch + offset, 0, 127)

        return self if inplace else NoteSequence(notes, copy.deepcopy(self.sustain_periods))

    def to_event_sequence(self, time_step_increment=10, max_time_steps=100, velocity_bins=32,
                          sustain_period_encode_mode=SustainPeriodEncodeMode.EVENTS, clean=True):
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
        :param sustain_period_encode_mode:
            The way in which sustain periods should be encoded. Defaults to :attr:`NoteSequence.SustainPeriodEncodeMode.EVENTS`.

            * If set to :attr:`NoteSequence.SustainPeriodEncodeMode.NONE`, sustain periods will be ignored.
            * If set to :attr:`NoteSequence.SustainPeriodEncodeMode.EXTEND`, notes within sustain periods will be extended
              until the end of the period or to the start of the next note of the same pitch, whichever comes first.
            * If set to :attr:`NoteSequence.SustainPeriodEncodeMode.EVENTS`, sustain periods will be encoded as events.
        :param clean:
            Indicates whether the event sequence should be cleaned up after conversion. This will remove undesirable events
            such as time shifts with value equal to zero or note on events followed directly by note off events.
        :returns:
            A list of :class:`Event` objects.

        '''

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
        # 
        # The same issue is present with sustain periods. Both notes and sustain periods are events
        # that have duration. They are referred to as "marker objects" since their corresponding events
        # mark whether they are on or off.

        #  We split each marker into two separate events: ON and OFF. We construct a MarkerInfo
        # object for each marker which contains the event type of the marker along with the time
        # that the marker occurs and, if necessary, the object that the marker refers to.
        class MarkerInfo:
            def __init__(self, marker_type, active, time, data=None):
                '''
                Initializes an instance of :class:`MarkerInfo`.

                :param marker_type:
                    A string indicating the type of the marker.
                :param active:
                    A boolean indicating whether this marker represents an ON or OFF event.
                    If true, it represents ON; otherwise, it represents OFF.
                :param time:
                    The time of the marker, in milliseconds, relative to the start of the sequence.
                :param data:
                    The object that the markers refers to. Defaults to ``None``.

                '''

                self.type = marker_type
                self.active = active
                self.time = time
                self.data = data

            def get_event_type(self, event_type_prefix=None):
                '''
                Gets the :class:`EventType` of this marker.

                :param event_type_prefix:
                    The prefix of the event type. If ``None``, it will default to :attr:`MarkerInfo.type`.

                '''

                prefix = self.type if event_type_prefix is None else event_type_prefix
                return EventType['{}_{}'.format(prefix, 'ON' if self.active else 'OFF')]

        markers = []

        # Make sure that the notes are in sorted order.
        # If they aren't then our events will also be in the wrong order!
        ordered_notes = sorted(self.notes, key=lambda x: x.start)
        ordered_sustain_periods = sorted(self.sustain_periods, key=lambda x: x.start)

        # We only need to add sustain markers if we are encoding them as events.
        if sustain_period_encode_mode == NoteSequence.SustainPeriodEncodeMode.EVENTS:
            for sustain_period in ordered_sustain_periods:
                markers.extend([
                    MarkerInfo('SUSTAIN', True, sustain_period.start, sustain_period),
                    MarkerInfo('SUSTAIN', False, sustain_period.end, sustain_period)
                ])
        elif sustain_period_encode_mode == NoteSequence.SustainPeriodEncodeMode.EXTEND:
            start_note_index = 0
            for sustain_period in ordered_sustain_periods:
                notes_in_interval = []
                for i in range(start_note_index, len(ordered_notes)):
                    note = ordered_notes[i]

                    if note.start < sustain_period.start: continue
                    if note.start > sustain_period.end:
                        break
                
                    notes_in_interval.append(note)

                if len(notes_in_interval) > 0:
                    start_note_index = i
                    
                    # The start time of previous notes by pitch.
                    previous_note_start_times = {}
                    for note in reversed(notes_in_interval):
                        if note.pitch in previous_note_start_times:
                            note.end = previous_note_start_times[note.pitch]
                        else:
                            note.end = max(sustain_period.end, note.end)
                        previous_note_start_times[note.pitch] = note.start
        
        for note in ordered_notes:
            markers.extend([
                MarkerInfo('NOTE', True, note.start, note),
                MarkerInfo('NOTE', False, note.end, note)
            ])

        # Sort event markers by time
        markers.sort(key=lambda x: x.time)

        events = []
        current_time = 0
        current_velocity = 0
        for marker in markers:
            # The interval of time between current time and the event, in time steps.
            interval = int(round(marker.time - current_time) / time_step_increment)
            if max_time_steps is not None:
                # If the interval of time exceeds the max time steps, we need to break it up into
                # multiple time shift events...
                for i in range(interval // max_time_steps):
                    events.append(Event(EventType.TIME_SHIFT, max_time_steps))

                # Get the remaining time and if it isn't zero, add the time shift command.
                interval %= max_time_steps

            if interval > 0:
                events.append(Event(EventType.TIME_SHIFT, interval))

            if marker.type == 'NOTE': 
                note = marker.data

                # If the note velocity is different from our current velocity,
                # we add a velocity event to indicate a change in velocity.
                if current_velocity != note.velocity:
                    # Scale the velocity value so that it is in a velocity bin.
                    # MIDI velocity ranges from 0 to 127.
                    # 
                    # The velocity bins are zero-indexed. Therefore, if we have
                    # four bins: 0-31, 32-63, 64-95, 96-127, they are numbered
                    # 0, 1, 2, and 3 respectively.
                    velocity_bin = (note.velocity * velocity_bins) // 128
                    events.append(Event(EventType.VELOCITY, velocity_bin))
        
                # Add note ON/OFF event
                events.append(Event(marker.get_event_type(), note.pitch))
                current_velocity = note.velocity
            elif marker.type == 'SUSTAIN':
                events.append(Event(marker.get_event_type(), None))

            current_time = marker.time

        if clean:
            # A list containing the index of the items to remove.
            remove_queue = []
            for i in range(len(events) - 1, -1, -1):
                event = events[i]
                if event.type == EventType.TIME_SHIFT and event.value == 0:
                    remove_queue.append(i)

                # If the current event is a NOTE_OFF event, check if the one preceeding it
                # was a NOTE_ON event (or vice-versa) If it is, we should remove both events 
                # since it is useless to turn on a note just to turn it off right after
                # (or to turn off a note just to turn it back on).
                if event.type == EventType.NOTE_OFF and i - 1 >= 0 and events[i - 1].type == EventType.NOTE_ON or \
                   event.type == EventType.NOTE_ON and i - 1 >= 0 and events[i - 1].type == EventType.NOTE_OFF:
                    # Make sure the notes are of the same pitch!
                    # It's perfectly okay (and normal!) for there to be a NOTE_ON and NOTE_OFF (or vice-versa) directly 
                    # after one another so long as these events are referencing different notes. It is only a problem 
                    # when we are turning off a note with the same pitch directly after turning it on...
                    if event.value == events[i - 1].value:
                        remove_queue.append(i)
                        remove_queue.append(i - 1)

            remove_queue.sort(reverse=True)
            for i in remove_queue:
                events.pop(i)

        return EventSequence(events, time_step_increment, max_time_steps, velocity_bins)

    def to_midi(self, filepath, program=1):
        '''
        Creates a MIDI file from this :class:`NoteSequence`.

        :param filepath:
            The path to the MIDI output file.
        :param programs:
            The MIDI program (instrument sound) to use. Defaults to 1 (Acoustic Grand Piano).
            See https://en.wikipedia.org/wiki/General_MIDI#Parameter_interpretations for a full
            list of all program numbers and their corresponding instruments.
    
        '''

        midi = PrettyMIDI()
        instrument = Instrument(program=program)
        for note in self.notes:
            # PrettyMIDI timing is in seconds so we need to convert our start and times to seconds.
            midi_note = MIDINote(note.velocity, note.pitch, note.start / 1000, note.end / 1000)
            instrument.notes.append(midi_note)

        for sustain_period in self.sustain_periods:
            # Control number 64 indicates Damper Pedal on/off (Sustain).
            # A value that is:
            #   * less than or equal to 63 indicates OFF
            #   * greater than or equal to 64 indicates ON

            instrument.control_changes.append(ControlChange(64, 64, sustain_period.start / 1000))
            instrument.control_changes.append(ControlChange(64, 63, sustain_period.end / 1000))

        midi.instruments.append(instrument)
        midi.write(filepath)
        
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
            sustains = []
            for instrument in midi.instruments:
                if ignore_drums and instrument.is_drum: continue
                if programs is not None and not instrument.program in programs: continue

                for midi_note in instrument.notes:
                    # PrettyMIDI timing is in seconds so we have to convert them to milliseconds.
                    notes.append(Note(midi_note.start * 1000, midi_note.end * 1000, midi_note.pitch, midi_note.velocity))

                # Get all the times that a sustain control (control number 64) changes.
                controls = [x for x in instrument.control_changes if x.number == 64]
                current_sustain_period = None
                for control in controls:
                    # Convert timing of the controls to milliseconds
                    control.time *= 1000

                    # If there is no sustain period currently and the sustain is down (control value >= 64),
                    # then this is the start of a new sustain period.
                    if control.value >= 64 and current_sustain_period is None:
                        current_sustain_period = SustainPeriod(control.time, None)
                    # If control.value is less than 64, the sustain pedal has been released.
                    elif control.value < 64:
                        if current_sustain_period is not None:
                            current_sustain_period.end = control.time
                            sustains.append(current_sustain_period)
                            current_sustain_period = None
                        elif len(sustains) > 0:
                            # If the sustain pedal has been released but there is no current sustain period,
                            # that must mean that the previous sustain period has been extended.
                            sustains[-1].end = control.time
                
            return NoteSequence(notes, sustains)

class EventSequence:
    '''
    The event-based representation of a :class:`NoteSequence`. 

    :ivar events:
        A list of :class:`Event` objects to add to the sequence.
    :ivar time_step_increment:
        The number of milliseconds that a single step in time represents.
    :ivar max_time_steps:
        The maximum number of time steps that a single event can shift time by.
    :ivar velocity_bins:
        The number of bins to quantize t he note velocity values into.
    
    '''

    def __init__(self, events, time_step_increment, max_time_steps, velocity_bins):
        '''
        Initialize an instance of :class:`EventSequence`.

        :param events:
            A list of :class:`Event` objects to add to the sequence.
        :param time_step_increment:
            The number of milliseconds that a single step in time represents.
        :param max_time_steps:
            The maximum number of time steps that a single event can shift time by.
            If this is ``None``, there is no limit.
        :param velocity_bins:
            The number of bins to quantize the note velocity values into.

        '''

        self.events = events
        self.time_step_increment = time_step_increment
        self.max_time_steps = max_time_steps
        self.velocity_bins = velocity_bins

    def to_one_hot_encoding(self):
        '''
        Encodes this :class:`EventSequence` as a series of one-hot encoded events

        :returns:
            An instance of :class:`OneHotEncodedEventSequence`.
        '''

        return OneHotEncodedEventSequence.encode(self)

    def to_integer_encoding(self):
        '''
        Encodes this :class:`EventSequence` as series of integers.

        :returns:
            An instance of :class:`IntegerEncodedEventSequence`.

        '''

        return IntegerEncodedEventSequence.encode(self)

    @staticmethod
    def _compute_event_value_ranges(time_step_increment, max_time_steps, velocity_bins):
        '''
        Calculuates the event value ranges

        '''

        value_ranges = collections.OrderedDict()

        # NOTE_ON and NOTE_OFF take a MIDI pitch value which ranges from 0 to 127.
        value_ranges[EventType.NOTE_ON] = range(0, 128)
        value_ranges[EventType.NOTE_OFF] = range(0, 128)

        # VELOCITY takes a MIDI velocity which ranges from 0 to the number of velocity bins - 1.
        # This is because velocity bins are zero-indexed.
        value_ranges[EventType.VELOCITY] = range(0, velocity_bins)
        
        max_time_steps = max_time_steps

        # Time shift doesn't accept zero since it is useless to do a time shift by no time steps.
        value_ranges[EventType.TIME_SHIFT] = range(1, max_time_steps + 1)

        # SUSTAIN events simply marker the start/end of a period.
        # They have no parameters...
        value_ranges[EventType.SUSTAIN_ON] = None
        value_ranges[EventType.SUSTAIN_OFF] = None

        return value_ranges

    @property
    def event_value_ranges(self):
        '''
        Gets the range of values for each :class:`EventType`.

        :note:
            If a range is `None`, it means that the event type does not accept any parameter values.
        :returns:
            A :class:`collections.OrderedDict` which maps :class:`EventType` to a
            :class:`range` object representing the range of values for each event type.

        '''

        # If no max time step value is given (i.e. it is None), we just get the largest time shift value in the event sequence.
        max_time_steps = self.max_time_steps if self.max_time_steps is not None else \
            max(event.value for event in self.events if event.type == EventType.TIME_SHIFT)

        return EventSequence._compute_event_value_ranges(
            self.time_step_increment,
            max_time_steps,
            self.velocity_bins
        )

    @staticmethod
    def _compute_event_dimensions(event_value_ranges):
        '''
        Computes the event dimensions.

        '''

        dimensions = collections.OrderedDict()
        for event_type, value_range in event_value_ranges.items():
            if value_range is None:
                value_range = range(0, 0)
            
            dimensions[event_type] = value_range.stop - value_range.start

        return dimensions

    @property
    def event_dimensions(self):
        '''
        Gets the dimension of each :class:`EventType`.

        :note:
            The dimension refers to the length of the range of values that each type of event
            accepts as parameters. If the dimension is zero, this means that the
            event does not accept any values (i.e. :attr:`Event.value` is ``None``).

        :returns:
            A :class:`collections.OrderedDict` which maps :class:`EventType` to integers
            representing the dimension of each event type.
            
        '''

        return EventSequence._compute_event_dimensions(self.event_value_ranges)

    @staticmethod
    def _compute_event_ranges(event_dimensions):
        '''
        Computes event ranges.

        '''

        offset = 0
        ranges = collections.OrderedDict()
        for event_type, dimension in event_dimensions.items():
            # If the dimension is zero, this means that the event has no parameter values.
            # Therefore, the event merely acts a boolean: it is either on or off.
            # However, we still require one element to encode this state.
            if dimension == 0:
                dimension += 1

            ranges[event_type] = range(offset, offset + dimension)
            offset += dimension
        
        return ranges

    @property
    def event_ranges(self):
        '''
        Gets the range of each event type in the one-hot encoded vector.

        :note:
            Suppose our event sequence consists of two events: ON and OFF,
            which both have value ranges from 1 to 4. Therefore, our one-
            hot encoded vector will have the form: [0, 0, 0, 0 | 0, 0, 0, 0]
            where the line indicates a new event type (separating event ON and OFF).

            The index range for the ON command is [0, 3] and the index range for the
            OFF command is [4, 7]. This function computes these ranges.
        :returns:
            A :class:`collections.OrderedDict` which maps :class:`EventType` to a 
            :class:`range` object representing the range of the event type.

        '''

        return self._compute_event_ranges(self.event_dimensions)

    def to_note_sequence(self):
        '''
        Converts this :class:`EventSequence` to a :class:`NoteSequence`.

        '''

        # The current time, in milliseconds.
        current_time = 0
        current_velocity = 0
        
        # The current notes mapped by pitch.
        current_notes = {}
        current_sustain_period = None

        notes = []
        sustain_periods = []
        for event in self.events:
            if event.type == EventType.NOTE_ON:
                # Check if the note is already on. In this case, we skip this event.
                if event.value in current_notes and current_notes[event.value] is not None: continue

                # We use zero as a placeholder value here since we don't know the end time yet.
                current_notes[event.value] = Note(current_time, 0, event.value, current_velocity)
            elif event.type == EventType.NOTE_OFF:
                # Check if the note is already off. In this case, we skip this event.
                if event.value not in current_notes or current_notes[event.value] is None: continue

                note = current_notes[event.value]
                note.end = current_time
                notes.append(note)

                # Clear the entry for this note in the current notes dictionary since the note is no longer active.
                current_notes[event.value] = None
            elif event.type == EventType.TIME_SHIFT:
                # Time shift value is in time steps so we need to convert to milliseconds.
                # Time step increment is the number of milliseconds forward in one time step.
                current_time += event.value * self.time_step_increment
            elif event.type == EventType.VELOCITY:
                # Convert the velocity bin index back to a MIDI velocity.
                # Of course, since we are quantizing these values, we will lose some precision.
                current_velocity = (128 * event.value) // self.velocity_bins 
            elif event.type == EventType.SUSTAIN_ON:
                # Make sure we don't already have a sustain period active.
                if current_sustain_period is not None: continue

                # We use zero as a placeholder value here since we don't know the end time yet.
                current_sustain_period = SustainPeriod(current_time, 0)
            elif event.type == EventType.SUSTAIN_OFF:
                # Make sure a sustain period is actually active.
                if current_sustain_period is None: continue

                current_sustain_period.end = current_time
                sustain_periods.append(current_sustain_period)

                # Clear the current sustain period since it is no longer active.
                current_sustain_period = None

        return NoteSequence(notes, sustain_periods)

    @staticmethod
    def from_file(filepath, decode=True):
        '''
        Creates an :class:`EventSequence` from an :class:`EncodedEventSequence` serialized as a file.

        :param filepath:
            The path to the file containing the encoded event sequence.
        :param decode:
            Indicates whether the file should be decoded. If ``True``, it is decoded into an :class:`EventSequence`.
            Otherwise, it is left encoded as an :class:`EncodedEventSequence`.
        :returns:
            An instance of :class:`EventSequence` if ``decode`` is ``True``; otherwise, returns an instance of  :class:`EncodedEventSequence`.

        '''

        _ENCODING_TYPE_MAP = {
            OneHotEncodedEventSequence.get_encoding_type(): OneHotEncodedEventSequence,
            IntegerEncodedEventSequence.get_encoding_type(): IntegerEncodedEventSequence 
        }

        filepath = Path(filepath)
        with open(filepath, 'rb') as file:
            encoding_type_id = _load_encoding_type_id(file)
            if encoding_type_id not in _ENCODING_TYPE_MAP:
                raise InvalidEncodingTypeError('Cannot load \'{}\' as an EventSequence! \'{}\' is not a valid encoding type id.' \
                                                .format(filepath, encoding_type_id))
        
        return _ENCODING_TYPE_MAP[encoding_type_id].from_file(filepath, decode=decode)

    def __repr__(self):
        return '\n'.join(str(event) for event in self.events)

class EncodedEventSequence(abc.ABC):
    '''
    The base class for all encoded event sequences.

    '''

    # The format of the encoding type id (unsigned long long)
    _ENCODING_TYPE_ID_FORMAT = 'Q'

    @abc.abstractstaticmethod
    def encode(event_sequence):
        '''
        Encodes the specified :class:`EventSequence`.

        :returns:
            An instance of :class:`EncodedEventSequence`.

        '''

        pass

    @abc.abstractmethod
    def decode(self):
        '''
        Decodes this :class:`EncodedEventSequence`. 
        
        :returns:
            An instance of :class:`EventSequence`.

        '''

        pass

    @abc.abstractmethod
    def to_file(self, filepath):
        '''
        Writes this :class:`EncodedEventSequence` to the specified filepath.

        :param filepath:
            The destination of the encoded sequence.
        
        '''

        pass

    @abc.abstractstaticmethod
    def from_file(filepath, decode=False):
        '''
        Loads a :class:`EncodedEventSequence` from the specified filepath.

        :param filepath:
            The source of the encoded sequence.
        :param decode:
            Indicates whether the file should be decoded into an :class:`EventSequence`.
            Defaults to ``False``.
        :returns:
            An instance of :class:`EventSequence` if ``decode`` is ``True``;
            othwerwise, an instance of :class:`EncodedEventSequence`.

        '''

        pass

    @abc.abstractstaticmethod
    def get_encoding_type():
        '''
        Gets the unique identifier for this type of :class:`EncodedEventSequence`.

        :note:
            The encoding type id is the first integer in all serialized :class:`EncodedEventSequence`.
            It allows the decoder to infer the encoding type at runtime without having to read the format.

        :returns:
            An integer value representing the encoding type.
        '''

        pass

def _load_encoding_type_id(file):
    '''
    Loads an encoding type id header from specfiied file object.
    
    :param file:
        A file-like object.

    '''

    pack_format = EncodedEventSequence._ENCODING_TYPE_ID_FORMAT
    encoding_type_id, = struct.unpack(pack_format, file.read(struct.calcsize(pack_format)))

    return encoding_type_id

class MismatchedOneHotVectorError(Exception):
    '''
    Raised when a :class:`OneHotEncodedEventSequence` has
    mismatched one-hot vectors (i.e. different shapes).

    '''

    pass

class InvalidEncodingTypeError(Exception):
    '''
    Raised when an :class:`EncodedEventSequence` has an invalid
    encoding type header.

    '''

    pass

class OneHotEncodedEventSequence(EncodedEventSequence):
    '''
    A one-hot encoded representation of an :class:`EventSequence`.

    :ivar time_step_increment:
        The number of milliseconds that a single step in time represents.
    :ivar event_ranges:
        The range of each event type in the one-hot encoded vector.
    :ivar event_value_ranges:
        The range of values for each :class:`EventType`.
    :ivar vectors:
        A list of one-hot encoded vectors representing the events.

    '''

    # The type of the event, start, and stop. 
    _EVENT_RANGE_FORMAT = 'hhh'

    # The type of the event, start, and stop.
    _EVENT_VALUE_RANGE_FORMAT = 'hhh'

    def __init__(self, time_step_increment, event_ranges, event_value_ranges, vectors=None):
        '''
        Initializes an instance of :class:`OneHotEncodedEventSequence`.

        :param time_step_increment:
            The number of milliseconds that a single step in time represents.
        :param event_ranges:
            The range of each event type in the one-hot encoded vector.
        :param event_value_ranges:
            The range of values for each :class:`EventType`.
        :param vectors:
            A list of one-hot encoded vectors representing the events.

        '''

        self.event_ranges = event_ranges
        self.event_value_ranges = event_value_ranges
        self.time_step_increment = time_step_increment
        self.vectors = vectors if vectors is not None else list()

    @property
    def one_hot_size(self):
        '''
        Gets the size of the one-hot vector that is needed to encode 
        an :class:`Event` with the specified event ranges.

        '''

        return self.__class__.get_one_hot_size(self.event_ranges)

    @staticmethod
    def get_one_hot_size(event_ranges):
        '''
        Gets the size of the one-hot vector that is needed to encode 
        an :class:`Event` with the specified event ranges.

        :param event_ranges:
            The range of each event type in the one-hot encoded vector.

        '''

        return event_ranges[next(reversed(event_ranges))].stop

    @classmethod
    def encode(cls, event_sequence):
        '''
        Encodes an :class:`EventSequence` as a series of one-hot encoded events.

        :param event_sequence:
            The :class:`EventSequence` to encode.
        :returns:
            An instance of :class:`OneHotEncodedEventSequence`.

        '''

        event_ranges = event_sequence.event_ranges
        one_hot_size = cls.get_one_hot_size(event_ranges)
        
        len_events = len(event_sequence.events)
        vectors = [None] * len_events
        event_value_ranges = event_sequence.event_value_ranges
        for i in range(len_events):
            event = event_sequence.events[i]

            index_offset = 0
            if event.value is not None:
                index_offset = event.value - event_value_ranges[event.type].start

            vectors[i] = [0] * one_hot_size
            vectors[i][event_ranges[event.type].start + index_offset] = 1

        return cls(event_sequence.time_step_increment, event_ranges, event_value_ranges, vectors)

    def decode(self):
        '''
        Decodes this :class:`OneHotEncodedEventSequence`. 
        
        :returns:
            An instance of :class:`EventSequence`.

        '''

        if not all(len(vector) == len(self.vectors[0]) for vector in self.vectors):
            raise MismatchedOneHotVectorError()

        events = []
        for vector in self.vectors:
            hot_index = vector.index(1)
            for event_type, event_range in self.event_ranges.items():
                if hot_index in event_range: break

            value = None
            if self.event_value_ranges[event_type] is not None:
                value = hot_index - event_range.start + self.event_value_ranges[event_type].start

            events.append(Event(event_type, value))
        
        # The max time steps value is the largest time step value that
        # the time shift event accepts. Therefore, we can use this range
        # to find the max_time_steps value.
        max_time_steps = self.event_value_ranges[EventType.TIME_SHIFT].stop

        # The velocity event value ranges from 0 to the velocity bin count.
        # Thus, we can use this range to find the velocity_bins value.
        velocity_bins = self.event_value_ranges[EventType.VELOCITY].stop

        return EventSequence(events, self.time_step_increment, max_time_steps, velocity_bins)

    def to_file(self, filepath):
        '''
        Writes this :class:`OneHotEncodedEventSequence` to the specified filepath.

        :param filepath:
            The destination of the encoded sequence.
        
        '''

        with open(filepath, 'wb+') as file:
            header_format = self._get_header_format()

            # Write the header
            unpacked_event_ranges = list(itertools.chain(*[(int(_type), _range.start, _range.stop) for _type, _range in self.event_ranges.items()]))
            unpacked_event_value_ranges = list(itertools.chain(*[(int(_type), _range.start if _range is not None else -1, \
                _range.stop if _range is not None else -1) for _type, _range in self.event_value_ranges.items()]))

            file.write(struct.pack(EncodedEventSequence._ENCODING_TYPE_ID_FORMAT + header_format, 
                self.__class__.get_encoding_type(),
                len(self.event_ranges), *unpacked_event_ranges,
                len(self.event_value_ranges), *unpacked_event_value_ranges,
                self.time_step_increment
            ))

            one_hot_vector_format = self.__class__._get_one_hot_vector_format(self.one_hot_size)
            for vector in self.vectors:
                file.write(struct.pack(one_hot_vector_format, *vector))

    def _get_header_format(self):
        '''
        Gets the format of the header.

        '''

        # The first integer indicates how many event ranges/event values ranges to read.
        event_range_format = 'i' + self.__class__._EVENT_RANGE_FORMAT * len(self.event_ranges)
        event_value_range_format = 'i' + self.__class__._EVENT_VALUE_RANGE_FORMAT * len(self.event_value_ranges)
        
        # The final integer is for the time step increment.
        return event_range_format + event_value_range_format + 'h'

    @staticmethod
    def _get_one_hot_vector_format(one_hot_vector_size):
        '''
        Gets the format of a one-hot vector.

        '''

        return '?' * one_hot_vector_size

    @classmethod
    def from_file(cls, filepath, decode=False):
        '''
        Loads a :class:`OneHotEncodedEventSequence` from the specified filepath.

        :param filepath:
            The source of the encoded sequence.
        :param decode:
            Indicates whether the :class:`OneHotEncodedEventSequence` file should be 
            directly decoded to an event sequence. Defaults to ``False``.
        :returns:
            An instance of :class:`OneHotEncodedEventSequence`.

        '''

        with open(filepath, 'rb') as file:
            # Read the encoding type id
            encoding_type_id = _load_encoding_type_id(file)
            if encoding_type_id != cls.get_encoding_type():
                raise InvalidEncodingTypeError('Cannot decode \'{}\' as OneHotEncodedEventSequence since the ' + 
                                               'encoding type id header does not match.'.format(filepath))

            integer_size = struct.calcsize('i')
            header_size = struct.calcsize(EncodedEventSequence._ENCODING_TYPE_ID_FORMAT)

            # Load the event ranges
            event_ranges_len, = struct.unpack('i', file.read(integer_size))
            event_ranges = collections.OrderedDict()
            event_range_size = struct.calcsize(cls._EVENT_RANGE_FORMAT)
            for i in range(event_ranges_len):
                event_type, start, stop = struct.unpack(cls._EVENT_RANGE_FORMAT, file.read(event_range_size))
                event_ranges[_EVENT_TYPE_MAPPINGS[event_type]] = range(start, stop)

            header_size += integer_size + event_range_size * event_ranges_len     

            # Load the event value ranges
            event_value_ranges_len, = struct.unpack('i', file.read(integer_size))
            event_value_ranges = collections.OrderedDict()
            event_value_range_size = struct.calcsize(cls._EVENT_VALUE_RANGE_FORMAT)
            for i in range(event_value_ranges_len):
                event_type, start, stop = struct.unpack(cls._EVENT_VALUE_RANGE_FORMAT, file.read(event_value_range_size))
                value_range = None

                # This is a sort of hack: if both start and stop are -1, the range is None.
                if start != -1 and stop != -1:
                    value_range = range(start, stop)

                event_value_ranges[_EVENT_TYPE_MAPPINGS[event_type]] = value_range

            header_size += integer_size + event_value_range_size * event_value_ranges_len

            # Load the time step increment
            half_size = struct.calcsize('h')
            time_step_increment, = struct.unpack('h', file.read(half_size))
            header_size += half_size

            # Load remaining one-hot vectors
            buffer_length = os.stat(filepath).st_size - header_size
            # The number of elements that the one-hot vector has.
            one_hot_vector_length = cls.get_one_hot_size(event_ranges)
            one_hot_vector_format = cls._get_one_hot_vector_format(one_hot_vector_length)
            # The size, in bytes, of the one-hot vector.
            one_hot_vector_size = struct.calcsize(one_hot_vector_format)
            
            events = []
            for i in range(buffer_length // one_hot_vector_size):
                vector = struct.unpack(one_hot_vector_format, file.read(one_hot_vector_size))
                if decode:
                    events.append(cls.one_hot_vector_as_event(vector, event_ranges, event_value_ranges))
                else:
                    events.append(list(map(int, vector)))

            if decode:
                # The max time steps value is the largest time step value that
                # the time shift event accepts. Therefore, we can use this range
                # to find the max_time_steps value.
                max_time_steps = event_value_ranges[EventType.TIME_SHIFT].stop

                # The velocity event value ranges from 0 to the velocity bin count.
                # Thus, we can use this range to find the velocity_bins value.
                velocity_bins = event_value_ranges[EventType.VELOCITY].stop
                return EventSequence(events, time_step_increment, max_time_steps, velocity_bins)
            else:
                return OneHotEncodedEventSequence(time_step_increment, event_ranges, event_value_ranges, events)

    def get_encoding_type():
        '''
        Gets the unique identifier for this type of :class:`OneHotEncodedEventSequence`.

        :note:
            The encoding type id is the first integer in all serialized :class:`OneHotEncodedEventSequence`.
            It allows the decoder to infer the encoding type at runtime without having to read the format.

        :returns:
            An integer value representing the encoding type.
        '''

        return 9223372036854775806

    @classmethod
    def event_as_one_hot_vector(cls, event, event_ranges, event_value_ranges, as_numpy_array=False, numpy_dtype=np.int):
        '''
        Converts an :class:`Event` to a one-hot vector.

        :param event:
            The :class:`Event` to convert.
        :param event_ranges:
            The range of each event type in the one-hot encoded vector.
        :param event_value_ranges:
            The range of values for each :class:`EventType`.
        :param as_numpy_array:
            Indicates whether a numpy array should be returned. Defaults to ``False``.
        :param numpy_dtype:
            The data type of the numpy array (if ``as_numpy_array`` is ``True``). Defaults to ``np.int``.
        :returns:
            An array-like object of integers (consisting of either 0 or 1) 
            representing the event as a one-hot vector.

        '''

        if as_numpy_array:
            vector = np.zeros(cls.get_one_hot_size(event_ranges), dtype=numpy_dtype)
        else:
            vector = [0] * cls.get_one_hot_size(event_ranges)

        index_offset = 0
        if event.value is not None:
            index_offset = event.value - event_value_ranges[event.type].start
        
        vector[event_ranges[event.type].start + index_offset] = 1
        return vector

    @staticmethod
    def one_hot_vector_as_event(vector, event_ranges, event_value_ranges):
        '''
        Converts a one-hot vector to an :class:`Event`

        :param vector:
            An array-like object of integers (consisting of either 0 or 1)
            representing the event as a one-hot vector.
        :param event_ranges:
            The range of each event type in the one-hot encoded vector.
        :param event_value_ranges:
            The range of values for each :class:`EventType`.
        :returns:
            An instance of :class:`Event`.

        '''

        if isinstance(vector, np.ndarray):
            # np.where returns a tuple of arrays, one for each dimension.
            # Vector has a single dimension and since it is a one-hot vector,
            # the array will consist of a single element: the index of the "hot" bit.
            hot_index = np.where(vector == 1)[0][0]
        else:
            if not isinstance(vector, list):
                vector = list(vector)
            
            hot_index = vector.index(1)
            
        for event_type, event_range in event_ranges.items():
            if hot_index in event_range: break

        value = None
        if event_value_ranges[event_type] is not None:
            value = hot_index - event_range.start + event_value_ranges[event_type].start

        return Event(event_type, value)

class IntegerEncodedEventSequence(EncodedEventSequence):
    '''
    A memory efficient and compact encoding for serializing :class:`EventSequence`
    objects to disk.
    
    This encoding consists of a series of integer ids that map the type and value.

    :note:
        This encoding consists of a list of two-dimensional integer tuples. Each
        event is encoded as a tuple containing the integer id of its type along
        with its value.

    :ivar time_step_increment:
        The number of milliseconds that a single step in time represents.
    :ivar max_time_steps:
        The maximum number of time steps that a single event can shift time by.
        If this is ``None``, there is no limit.
    :ivar velocity_bins:
        The number of bins to quantize the note velocity values into.
    :ivar events:
        A list of integer encoded events (two-dimensional integer tuples).


    '''

    _HEADER_FORMAT = 'hhh'
    _EVENT_FORMAT = 'hh'

    def __init__(self, time_step_increment, max_time_steps, velocity_bins, events=None):
        '''
        Initializes an instance of :class:`IntegerEncodedEventSequence`.

        :param time_step_increment:
            The number of milliseconds that a single step in time represents.
        :param max_time_steps:
            The maximum number of time steps that a single event can shift time by.
            If this is ``None``, there is no limit.
        :param velocity_bins:
            The number of bins to quantize the note velocity values into.
        :param events:
            A list of integer encoded events.

        '''

        self.time_step_increment = time_step_increment
        self.max_time_steps = max_time_steps
        self.velocity_bins = velocity_bins
        self.events = events if events is not None else list()

    @staticmethod
    def encode(event_sequence):
        '''
        Encodes an :class:`EventSequence`.

        :param event_sequence:
            The :class:`EventSequence` to encode.
        :returns:
            An instance of :class:`IntegerEncodedEventSequence`.

        '''

        events = []
        for event in event_sequence.events:
            events.append((int(event.type), Event.encode_value(event)))
        
        return IntegerEncodedEventSequence(event_sequence.time_step_increment, event_sequence.max_time_steps,
                                           event_sequence.velocity_bins, events)

    def decode(self):
        '''
        Decodes this :class:`IntegerEncodedEventSequence`. 
        
        :returns:
            An instance of :class:`EventSequence`.

        '''

        events = []
        for encoded_event in self.events:
            event_type, value = encoded_event
            events.append(Event(_EVENT_TYPE_MAPPINGS[event_type], Event.decode_value(value)))

        return EventSequence(events, self.time_step_increment, self.max_time_steps, self.velocity_bins)

    def to_file(self, filepath):
        '''
        Writes this :class:`IntegerEncodedEventSequence` to the specified filepath.

        :note:
            This will overwrite the file if it already exists.

        :param filepath:
            The destination of the encoded sequence.
        
        '''

        with open(filepath, 'wb+') as file:
            # Each event is encoded as an integer tuple.
            events_format = self.__class__._EVENT_FORMAT * len(self.events)

            # The first integer is for the encoding type id.
            # The next three integers are dedicated for describing the event sequence:
            #   Values: time step increments, max time steps, and velocity bins.
            encoded_sequence = struct.pack(EncodedEventSequence._ENCODING_TYPE_ID_FORMAT + 
                self.__class__._HEADER_FORMAT + events_format,
                self.__class__.get_encoding_type(), self.time_step_increment,
                self.max_time_steps, self.velocity_bins,
                *list(itertools.chain(*self.events))
            )
            
            file.write(encoded_sequence)

    @classmethod
    def _load_file_header(cls, file):
        '''
        Loads the header of an :class`IntegerEncodedEventSequence` file.

        :param file:
            A file-like object.
        :returns:
            The time step increment, max time steps, velocity bins, and header size (in bytes).

        '''

        # Read the encoding type id
        encoding_type_id = _load_encoding_type_id(file)
        if encoding_type_id != cls.get_encoding_type():
            raise InvalidEncodingTypeError('Cannot decode \'{}\' as IntegerEncodedEventSequence since the ' + 
                                           'encoding type id header does not match.'.format(filepath))

        header_size = struct.calcsize(cls._HEADER_FORMAT)
        time_step_increment, max_time_steps, velocity_bins = struct.unpack(cls._HEADER_FORMAT, file.read(header_size))
        # We have to add this since the header includes the encoding type id; however, we read it seperately so that we can
        # "short-circuit" the rest of the reading in case that the encoding type id doesn't match.
        header_size += struct.calcsize(EncodedEventSequence._ENCODING_TYPE_ID_FORMAT)

        return time_step_increment, max_time_steps, velocity_bins, header_size

    @classmethod
    def from_file(cls, filepath, decode=False):
        '''
        Loads an :class:`IntegerEncodedEventSequence` from the specified filepath.

        :param filepath:
            The source of the encoded sequence.
        :param decode:
            Indicates whether the file should be decoded into an :class:`EventSequence`.
            Defaults to ``False``.
        :returns:
            An instance of :class:`EventSequence` if ``decode`` is ``True``;
            othwerwise, it returns an instance of :class:`IntegerEncodedEventSequence`.

        '''

        with open(filepath, 'rb') as file:
            time_step_increment, max_time_steps, velocity_bins, header_size = cls._load_file_header(file)

            event_size = struct.calcsize(cls._EVENT_FORMAT)
            buffer_length = os.stat(filepath).st_size - header_size

            events = []
            for i in range(buffer_length // event_size):
                event_type, value = struct.unpack(cls._EVENT_FORMAT, file.read(event_size))
                if decode:
                    events.append(Event(_EVENT_TYPE_MAPPINGS[event_type], Event.decode_value(value)))
                else:
                    events.append((event_type, value))

            if decode:
                return EventSequence(events, time_step_increment, max_time_steps, velocity_bins)
            else:
                return IntegerEncodedEventSequence(time_step_increment, max_time_steps, velocity_bins, events)

    @staticmethod
    def event_to_id(event_type, event_value, event_ranges, event_value_ranges):
        '''
        Converts an event to an id.

        :param event_type:
            The :class:`EventType` of the event.
        :param event_value:
            The value of the event. Can be ``None``.
        :param event_ranges:
            A :class:`collections.OrderedDict` representing the range of 
            each event type in the one-hot encoded vector.
        :param event_value_ranges:
            A :class:`collections.OrderedDict` representing the range of 
            values for each :class:`EventType`.
        :returns:
            An integer id representing the event.

        '''

        offset = 0
        if event_value_ranges[event_type] is not None:
            offset = event_value - event_value_ranges[event_type].start
        return event_ranges[event_type].start + offset

    @staticmethod
    def id_to_event(event_id, event_ranges, event_value_ranges):
        '''
        Converts an integer id to an :class:`Event`.

        :param event_id:
            The integer id of the event.
        :param event_ranges:
            A :class:`collections.OrderedDict` representing the range of 
            each event type in the one-hot encoded vector.
        :param event_value_ranges:
            A :class:`collections.OrderedDict` representing the range of 
            values for each :class:`EventType`.
        :returns:
            An instance of :class:`Event`.

        '''

        for event_type, interval in event_ranges.items():
            if event_id in interval:
                # An event value range of type None means that the event has no value
                # (i.e. the value of the event is None itself).
                value = None
                if event_value_ranges[event_type] is not None:
                    value = event_id - interval.start + event_value_ranges[event_type].start
                
                return Event(event_type, value)

    @classmethod
    def event_ids_from_file(cls, filepath, as_numpy_array=False, numpy_dtype=np.int):
        '''
        Loads an :class:`IntegerEncodedEventSequence` from
        the specified filepath as single integer event ids.

        :note:
            These event ids act very similarly to one-hot vectors.
            They are simply to make data handling more efficient within the network.

        :param filepath:
            The source of the encoded sequence.
        :param as_numpy_array:
            Indicates whether the event ids should be returned as a numpy array.
            Defaults to ``False``.
        :param numpy_dtype:
            The data type of the numpy array (if ``as_numpy_array`` is ``True``). Defaults to ``np.int``.
        :returns:
            A list of integers representing the event ids, a ``collections.OrderedDict`` containing
            the event value ranges, a ``collections.OrderedDict`` representing the event ranges,
            and an additional three-dimensional tuple of integers representing the settings
            of the event sequence. It contains the time step increment, max time steps, 
            and velocity bins in that order.

        '''
        
        with open(filepath, 'rb') as file:
            time_step_increment, max_time_steps, velocity_bins, header_size = cls._load_file_header(file)

            # The size of a single encoded event, in bytes.
            event_size = struct.calcsize(cls._EVENT_FORMAT)
            buffer_length = os.stat(filepath).st_size - header_size

            # Compute the event ranges (used to create the event ids)
            event_value_ranges = EventSequence._compute_event_value_ranges(time_step_increment, max_time_steps, velocity_bins)
            event_ranges = EventSequence._compute_event_ranges(EventSequence._compute_event_dimensions(event_value_ranges))

            # The number of events to encode.
            event_count = buffer_length // event_size
            if as_numpy_array:
                event_ids = np.empty(event_count, dtype=numpy_dtype)
            else:
                event_ids = array.array('H')
            
            for i in range(event_count):
                event_type, value = struct.unpack(cls._EVENT_FORMAT, file.read(event_size))
                event_id = cls.event_to_id(_EVENT_TYPE_MAPPINGS[event_type], value, event_ranges, event_value_ranges)
                if as_numpy_array:
                    event_ids[i] = event_id
                else:
                    event_ids.append(event_id) 

            settings = (time_step_increment, max_time_steps, velocity_bins)
            return event_ids, event_value_ranges, event_ranges, settings

    @classmethod
    def event_ids_from_file_as_generator(cls, filepath):
        '''
        Loads an :class:`IntegerEncodedEventSequence` from
        the specified filepath as single integer event ids
        and returns them as a generator object.

        :note:
            These event ids act very similarly to one-hot vectors.
            They are simply to make data handling more efficient within the network.

        :param filepath:
            The source of the encoded sequence.
        :returns:
            A generator which yields integers representing the event ids.

        '''
        
        with open(filepath, 'rb') as file:
            time_step_increment, max_time_steps, velocity_bins, header_size = cls._load_file_header(file)

            # The size of a single encoded event, in bytes.
            event_size = struct.calcsize(cls._EVENT_FORMAT)
            buffer_length = os.stat(filepath).st_size - header_size

            # Compute the event ranges (used to create the event ids)
            event_value_ranges = EventSequence._compute_event_value_ranges(time_step_increment, max_time_steps, velocity_bins)
            event_ranges = EventSequence._compute_event_ranges(EventSequence._compute_event_dimensions(event_value_ranges))

            # The number of events to encode.
            event_count = buffer_length // event_size
            for i in range(event_count):
                event_type, value = struct.unpack(cls._EVENT_FORMAT, file.read(event_size))
                yield cls.event_to_id(_EVENT_TYPE_MAPPINGS[event_type], value, event_ranges, event_value_ranges)

    @classmethod
    def one_hot_from_file(cls, filepath, as_numpy_array=False, numpy_dtype=np.int):
        '''
        Loads an :class:`IntegerEncodedEventSequence` from
        the specified filepath as one-hot vectors.

        :note:
            These one-hot vectors are constructed based on the id of each event.
            An event id is fundmenetally similar to the one-hot vector representation. 
            The integer id of an event is the zero-based index of the  "hot" (active) bit 
            of its one-hot vector representation.
            
            These event ids act very similarly to one-hot vectors. They are simply to make 
            data handling more efficient within the network. However, we may often want
            to have access to the one-hot vectors directly.

        :param filepath:
            The source of the encoded sequence.
        :param as_numpy_array:
            Indicates whether the event ids should be returned as a numpy array.
            Defaults to ``False``.

            Note that numpy array creation is much faster than Python lists.
        :param numpy_dtype:
            The data type of the numpy array (if ``as_numpy_array`` is ``True``). Defaults to ``np.int``.
        :returns:
            A list of one-hot vectors, a ``collections.OrderedDict`` containing the event value 
            ranges, a ``collections.OrderedDict`` representing the event ranges, and an additional 
            three-dimensional tuple of integers representing the settings of the event sequence. 
            It contains the time step increment, max time steps, and velocity bins in that order.

        '''
        
        with open(filepath, 'rb') as file:
            time_step_increment, max_time_steps, velocity_bins, header_size = cls._load_file_header(file)

            # The size of a single encoded event, in bytes.
            event_size = struct.calcsize(cls._EVENT_FORMAT)
            buffer_length = os.stat(filepath).st_size - header_size

            # Compute the event ranges (used to create the event ids)
            event_value_ranges = EventSequence._compute_event_value_ranges(time_step_increment, max_time_steps, velocity_bins)
            event_ranges = EventSequence._compute_event_ranges(EventSequence._compute_event_dimensions(event_value_ranges))

            # The number of events to encode.
            event_count = buffer_length // event_size
            vector_dimensions = OneHotEncodedEventSequence.get_one_hot_size(event_ranges)
            if as_numpy_array:
                vectors = np.zeros((event_count, vector_dimensions), dtype=numpy_dtype)
            else:
                vectors = [[0] * vector_dimensions] * event_count
            
            for i in range(event_count):
                event_type, value = struct.unpack(cls._EVENT_FORMAT, file.read(event_size))
                event_id = cls.event_to_id(_EVENT_TYPE_MAPPINGS[event_type], value, event_ranges, event_value_ranges)
                
                # Set the "hot" bit of the one-hot vector.
                vectors[i][event_id] = 1

            settings = (time_step_increment, max_time_steps, velocity_bins)
            return vectors, event_value_ranges, event_ranges, settings

    @classmethod
    def one_hot_from_file_as_generator(cls, filepath, as_numpy_array=False, numpy_dtype=np.int):
        '''
        Loads an :class:`IntegerEncodedEventSequence` from
        the specified filepath as one-hot vectors and returns
        them as a generator object.

        :note:
            These one-hot vectors are constructed based on the id of each event.
            An event id is fundmenetally similar to the one-hot vector representation. 
            The integer id of an event is the zero-based index of the  "hot" (active) bit 
            of its one-hot vector representation.
            
            These event ids act very similarly to one-hot vectors. They are simply to make 
            data handling more efficient within the network. However, we may often want
            to have access to the one-hot vectors directly.

        :param filepath:
            The source of the encoded sequence.
        :param as_numpy_array:
            Indicates whether the one-hot vectors should be returned as a numpy array.
            Defaults to ``False``.

            Note that numpy array creation is much faster than Python lists.
        :param numpy_dtype:
            The data type of the numpy array (if ``as_numpy_array`` is ``True``). Defaults to ``np.int``.
        :returns:
            A generator which yields one-hot vectors representing the events.

        '''
        
        with open(filepath, 'rb') as file:
            time_step_increment, max_time_steps, velocity_bins, header_size = cls._load_file_header(file)

            # The size of a single encoded event, in bytes.
            event_size = struct.calcsize(cls._EVENT_FORMAT)
            buffer_length = os.stat(filepath).st_size - header_size

            # Compute the event ranges (used to create the event ids)
            event_value_ranges = EventSequence._compute_event_value_ranges(time_step_increment, max_time_steps, velocity_bins)
            event_ranges = EventSequence._compute_event_ranges(EventSequence._compute_event_dimensions(event_value_ranges))

            # The number of events to encode.
            event_count = buffer_length // event_size
            vector_dimensions = OneHotEncodedEventSequence.get_one_hot_size(event_ranges)
            if as_numpy_array:
                vectors = np.zeros((event_count, vector_dimensions), dtype=numpy_dtype)
            else:
                vectors = [[0] * vector_dimensions] * event_count
            
            for i in range(event_count):
                event_type, value = struct.unpack(cls._EVENT_FORMAT, file.read(event_size))
                event_id = cls.event_to_id(_EVENT_TYPE_MAPPINGS[event_type], value, event_ranges, event_value_ranges)
                
                # Set the "hot" bit of the one-hot vector.
                vectors[i][event_id] = 1
                yield vectors[i]

            return vectors

    def get_encoding_type():
        '''
        Gets the unique identifier for this type of :class:`IntegerEncodedEventSequence`.

        :note:
            The encoding type id is the first integer in all serialized :class:`IntegerEncodedEventSequence`.
            It allows the decoder to infer the encoding type at runtime without having to read the format.

        :returns:
            An integer value representing the encoding type.
        '''

        return 9223372036854775805