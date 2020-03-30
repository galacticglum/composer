'''
High-level data-structures for representing MIDI-like sequences of notes 
along with their associated event encodings.

'''

class Note:
    '''
    A representation of a musical note on a sequence.

    '''

    def __init__(self, start, end, pitch, velocity):
        '''
        Initializes a note.

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

    pass