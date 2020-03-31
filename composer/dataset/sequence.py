'''
High-level data-structures for representing MIDI-like sequences of notes 
along with their associated event encodings.

'''

import collections
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
                    # There are 128 possible velocity values: 0 to 127.
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

    def event_dimensions(self):
        '''
        Gets the dimension of each :class:`EventType`.

        :note:
            The dimension refers to the range of values that each type of event
            accepts as parameters. If the dimension is zero, this means that the
            event does not accept any values (i.e. :var:`Event.value` is ``None``).
        :returns:
            A :class:`collections.OrderedDict` which maps `EventType` to integers
            representing the dimension of each event type.
            
        '''

        dimensions = collections.OrderedDict()

        # NOTE_ON and NOTE_OFF take a MIDI pitch value which ranges from 0 to 127.
        dimensions[EventType.NOTE_ON] = 128
        dimensions[EventType.NOTE_OFF] = 128
        # VELOCITY takes a MIDI velocity which ranges from 0 to 127.
        dimensions[EventType.VELOCITY] = 128
        
        # If no max time step value is given (i.e. it is None), we just get the largest
        # time shift value in the event sequence.
        max_time_steps = self.max_time_steps if self.max_time_steps is not None else \
            max(event.value for event in self.events if event.type == EventType.TIME_SHIFT)

        dimensions[EventType.TIME_SHIFT] = max_time_steps

        # SUSTAIN events simply marker the start/end of a period.
        # They have no parameters...
        dimensions[EventType.SUSTAIN_ON] = 0
        dimensions[EventType.SUSTAIN_OFF] = 0

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
            A :class:`collections.OrderedDict` which maps `EventType` to a :class:`range`
            object representing the range of the event type.

        '''

        offset = 0
        ranges = collections.OrderedDict()
        for event_type, dimension in self.event_dimensions().items():
            ranges[event_type] = range(offset, offset + dimension)
            offset += dimension
        
        return ranges

    def __repr__(self):
        return '\n'.join(str(event) for event in self.events)