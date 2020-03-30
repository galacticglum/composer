'''
High-level data-structures for representing MIDI-like sequences of notes 
along with their associated event encodings.

'''

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
            The velocity of the note (as a MIDI velocity: from 1 to 127).
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
        The duration of the note.

        '''

        return self.end - self.start

class NoteSequence:
    '''
    A MIDI-like note sequence representation.

    '''

    def __init__(self, notes=None):
        '''
        Initializes an instance of class :class:`NoteSequence`.

        :param notes:
            If specified, these notes are added to the sequence.
        '''

        self.notes = []
        self.add(notes)

    def add(self, notes, maintain_order=True):
        '''
        Adds a list of notes to this :class:`NoteSequence`.

        :param notes:
            The :class:`Note` objects to add to this :class:`NoteSequence`.
        :param maintain_order:
            Indicates whether the internal representation of notes should be resorted so that
            it is in ascending order with respect to the starting time of each note. Defaults
            to ``True``.
        '''

        self.notes.extend(notes)
        if not maintain_order: return

        self.notes.sort(key=lambda x: x.start)

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