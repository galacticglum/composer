'''
High-level data-structures for representing MIDI-like sequences of notes 
along with their associated event encodings.

'''

import abc
import struct
import itertools
import collections
from enum import Enum, IntEnum
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

class EventType(IntEnum):
        '''
        The type of an :class:`Event`.
        
        '''

        NOTE_ON = 1
        NOTE_OFF = 2
        TIME_SHIFT = 3
        VELOCITY = 4
        SUSTAIN_ON = 5
        SUSTAIN_OFF = 6

class Event:
    '''
    An event representing some change on the NoteSequence.
    Every event has a type and value associated with it.

    '''

    # A reserved value for representing None as an integer.
    NONE_VALUE = -1

    def __init__(self, event_type, value):
        '''
        Initialize an instance of :class:`Event`.

        :param event_type:
            An :class:`EventType` value representing the type of the event.
        :param value:
            The value of the event. The type of this object depends on the 
            type of the event.

            :note:
                The value of an event is not restricted. It can be any object
                so long as the object can be represented as int. Also, when
                the event is decoded (after being encoded), the all values
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


    def __repr__(self):
        return 'Event(type={}, value={})'.format(str(self.type), self.value)

class SustainPeriod:
    '''
    A period of time where the sustain pedal is active.

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

    '''

    class SustainPeriodEncodeMode(Enum):
        '''
        The mode for encoding sustain periods.

        '''

        NONE = 1
        EXTEND = 2
        EVENTS = 3

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

    def to_event_sequence(self, time_step_increment=10, max_time_steps=100, velocity_bins=32,
                          sustain_period_encode_mode=SustainPeriodEncodeMode.EVENTS):
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
            The way in which sustain periods should be encoded. Defaults to :var:`NoteSequence.SustainPeriodEncodeMode.EVENTS`.

            * If set to :var:`NoteSequence.SustainPeriodEncodeMode.NONE`, sustain periods will be ignored.
            * If set to :var:`NoteSequence.SustainPeriodEncodeMode.EXTEND`, notes within sustain periods will be extended
              until the end of the period or to the start of the next note of the same pitch, whichever comes first.
            * If set to :var:`NoteSequence.SustainPeriodEncodeMode.EVENTS`, sustain periods will be encoded as events.
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
                    The prefix of the event type. If ``None``, it will default to :var:`MarkerInfo.type`.

                '''

                prefix = self.type if event_type_prefix is None else event_type_prefix
                return EventType['{}_{}'.format(prefix, 'ON' if self.active else 'OFF')]

        markers = []

        # We only need to add sustain markers if we are encoding them as events.
        if sustain_period_encode_mode == NoteSequence.SustainPeriodEncodeMode.EVENTS:
            ordered_sustain_periods = sorted(self.sustain_periods, key=lambda x: x.start)
            for sustain_period in ordered_sustain_periods:
                markers.extend([
                    MarkerInfo('SUSTAIN', True, sustain_period.start, sustain_period),
                    MarkerInfo('SUSTAIN', False, sustain_period.end, sustain_period)
                ])
        elif sustain_period_encode_mode == NoteSequence.SustainPeriodEncodeMode.EXTEND:
            # TODO: Implement sustain period extend encoding
            pass
        
        # Make sure that the notes are in sorted order.
        # If they aren't then our events will also be in the wrong order!
        ordered_notes = sorted(self.notes, key=lambda x: x.start)
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

        return EventSequence(events, time_step_increment, max_time_steps, velocity_bins)

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

    def event_value_ranges(self):
        '''
        Gets the range of values for each :class:`EventType`.

        :note:
            If a range is `None`, it means that the event type does not accept any parameter values.
        :returns:
            A :class:`collections.OrderedDict` which maps :class:`EventType` to a
            :class:`range` object representing the range of values for each event type.

        '''

        value_ranges = collections.OrderedDict()

        # NOTE_ON and NOTE_OFF take a MIDI pitch value which ranges from 0 to 127.
        value_ranges[EventType.NOTE_ON] = range(0, 128)
        value_ranges[EventType.NOTE_OFF] = range(0, 128)

        # VELOCITY takes a MIDI velocity which ranges from 0 to the number of velocity bins - 1.
        # This is because velocity bins are zero-indexed.
        value_ranges[EventType.VELOCITY] = range(0, self.velocity_bins)
        
        # If no max time step value is given (i.e. it is None), we just get the largest
        # time shift value in the event sequence.
        max_time_steps = self.max_time_steps if self.max_time_steps is not None else \
            max(event.value for event in self.events if event.type == EventType.TIME_SHIFT)

        # Time shift doesn't accept zero since it is useless to do a time shift by no time steps.
        value_ranges[EventType.TIME_SHIFT] = range(1, max_time_steps + 1)

        # SUSTAIN events simply marker the start/end of a period.
        # They have no parameters...
        value_ranges[EventType.SUSTAIN_ON] = None
        value_ranges[EventType.SUSTAIN_OFF] = None

        return value_ranges

    def event_dimensions(self):
        '''
        Gets the dimension of each :class:`EventType`.

        :note:
            The dimension refers to the length of the range of values that each type of event
            accepts as parameters. If the dimension is zero, this means that the
            event does not accept any values (i.e. :var:`Event.value` is ``None``).
        :returns:
            A :class:`collections.OrderedDict` which maps :class:`EventType` to integers
            representing the dimension of each event type.
            
        '''

        value_ranges = self.event_value_ranges()
        dimensions = collections.OrderedDict()

        for event_type, value_range in self.event_value_ranges().items():
            if value_range is None:
                value_range = range(0, 0)
            
            dimensions[event_type] = value_range.stop - value_range.start

        return dimensions

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

        offset = 0
        ranges = collections.OrderedDict()
        for event_type, dimension in self.event_dimensions().items():
            # If the dimension is zero, this means that the event has no parameter values.
            # Therefore, the event merely acts a boolean: it is either on or off.
            # However, we still require one element to encode this state.
            if dimension == 0:
                dimension += 1

            ranges[event_type] = range(offset, offset + dimension)
            offset += dimension
        
        return ranges

    def __repr__(self):
        return '\n'.join(str(event) for event in self.events)

class EncodedEventSequence(abc.ABC):
    '''
    The base class for all encoded event sequences.

    '''

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
    def from_file(filepath):
        '''
        Loads a :class:`EncodedEventSequence` from the specified filepath.

        :param filepath:
            The source of the encoded sequence.
        :returns:
            An instance of :class:`EncodedEventSequence`.

        '''

        pass

class MismatchedOneHotVectorError(Exception):
    '''
    Raised when a :class:`OneHotEncodedEventSequence` has
    mismatched one-hot vectors (i.e. different shapes).

    '''

    pass

class OneHotEncodedEventSequence(EncodedEventSequence):
    '''
    A one-hot encoded representation of an :class:`EventSequence`.

    '''

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
            A list of one-hot encoded vectors representing events.

        '''

        self.event_ranges = event_ranges
        self.event_value_ranges = event_value_ranges
        self.time_step_increment = time_step_increment
        self.vectors = vectors if vectors is not None else list()

    @staticmethod
    def encode(event_sequence):
        '''
        Encodes an :class:`EventSequence` as a series of one-hot encoded events.

        :param event_sequence:
            The :class:`EventSequence` to encode.
        :returns:
            An instance of :class:`OneHotEncodedEventSequence`.

        '''

        event_ranges = event_sequence.event_ranges()
        len_events = len(event_sequence.events)
        one_hot_size = event_ranges[next(reversed(event_ranges))].stop

        vectors = [None] * len_events
        event_value_ranges = event_sequence.event_value_ranges()
        for i in range(len_events):
            event = event_sequence.events[i]

            index_offset = 0
            if event.value is not None:
                index_offset = event.value - event_value_ranges[event.type].start

            vectors[i] = [0] * one_hot_size
            vectors[i][event_ranges[event.type].start + index_offset] = 1

        return OneHotEncodedEventSequence(event_sequence.time_step_increment, 
                                          event_ranges, event_value_ranges, vectors)

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

        raise NotImplementedError()

    @staticmethod
    def from_file(filepath):
        '''
        Loads a :class:`OneHotEncodedEventSequence` from the specified filepath.

        :param filepath:
            The source of the encoded sequence.
        :returns:
            An instance of :class:`OneHotEncodedEventSequence`.

        '''

        raise NotImplementedError()

class IntegerEncodedEventSequence(EncodedEventSequence):
    '''
    A memory efficient and compact encoding for serializing :class:`EventSequence`
    objects to disk.
    
    This encoding consists of a series of integer ids that map the type and value.

    :note:
        This encoding consists of a list of two-dimensional integer tuples. Each
        event is encoded as a tuple containing the integer id of its type along
        with its value.

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
            events.append(Event(EventType(event_type), Event.decode_value(value)))

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
            events_format = IntegerEncodedEventSequence._EVENT_FORMAT * len(self.events)

            # The first three integers are dedicated for describing the event sequence:
            #   Values: time step increments, max time steps, and velocity bins.
            encoded_sequence = struct.pack(IntegerEncodedEventSequence._HEADER_FORMAT + events_format, 
                self.time_step_increment, self.max_time_steps, self.velocity_bins,
                *list(itertools.chain(*self.events))
            )
            
            file.write(encoded_sequence)

    @staticmethod
    def from_file(filepath):
        '''
        Loads a :class:`IntegerEncodedEventSequence` from the specified filepath.

        :param filepath:
            The source of the encoded sequence.
        :returns:
            An instance of :class:`IntegerEncodedEventSequence`.

        '''

        with open(filepath, 'rb') as file:
            header_size = struct.calcsize(IntegerEncodedEventSequence._HEADER_FORMAT)
            time_step_increment, max_time_steps, velocity_bins = struct.unpack(IntegerEncodedEventSequence._HEADER_FORMAT, file.read(header_size))
            
            event_size = struct.calcsize(IntegerEncodedEventSequence._EVENT_FORMAT)
            buffer_length = Path(filepath).stat().st_size - header_size

            events = []
            for i in range(buffer_length // event_size):
                event_type, value = struct.unpack(IntegerEncodedEventSequence._EVENT_FORMAT, file.read(event_size))
                events.append((event_type, value))

            return IntegerEncodedEventSequence(time_step_increment, max_time_steps, velocity_bins, events)