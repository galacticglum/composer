'''
Utility methods.

'''

import logging
from tqdm import tqdm
from queue import Queue
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3, multithread=False, 
                     show_progress_bar=True, extend_result=False, initial_value=list()):
    '''
    A parallel version of the map function with a progress bar. 

    :note:
        This is a utility function for running parallel jobs with progress
        bar. Orignally from http://danshiebler.com/2016-09-14-parallel-progress-bar/.

        The implementation is identical to the source; however, the documentation and 
        code style has been modified to fit the style of this codebase.

    :param array:
        An array to iterate over.
    :param function:
        A python function to apply to the elements of array
    :param n_jobs:
        The number of cores to use. Defaults to 16.
    :param use_kwargs:
        Whether to consider the elements of array as dictionaries of 
        keyword arguments to function. Defaults to ``False``.
    :param front_num:
        The number of iterations to run serially before kicking off the
        parallel job. Useful for catching bugs
    :param multithread:
        If ``True``, a :class:``concurrent.futures.ThreadPoolExecutor`` will be used rather than a :class:``concurrent.futures.ProcessPoolExecutor``.
        Defaults to ``False``.
    :param show_progress_bar:
        Indicates whether a loading progress bar should be displayed while the process runs.
        Defaults to ``True``.
    :param extend_result:
        Indicates whether the resultant list should be extended rather than appended to. Defaults to ``False``.

        Note that this requires that the return value of ``function`` is an array-like object.
    :param initial_value:
        The initial value of the resultant array. This should be an array-like object.
    :returns:
        A list of the form [function(array[0]), function(array[1]), ...].

    '''
    # We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]

    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]

    # Assemble the workers
    pool_type = ThreadPoolExecutor if multithread else ProcessPoolExecutor
    with pool_type(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]

        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True,
            'disable': not show_progress_bar
        }

        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs): pass

    out = initial_value
    out.extend(front)
    
    # Get the results from the futures. 
    _add_func = lambda x: out.extend(x) if extend_result else out.append(x)
    for i, future in tqdm(enumerate(futures)):
        try:
            _add_func(future.result())
        except Exception as e:
            _add_func(e)

    return out

class ObjectPool:
    '''
    Generic object pool manager.

    '''

    def __init__(self, create_func, name=None, warm_stride_size=1):
        '''
        Initializes an instance of :class:`ObjectPool`.

        :param create_func:
            A parameterless function creates the objects stored in the pool.
        :param warm_stride_size:
            The number of objects to create when the object pool becomes empty.
            Defaults to 1.

        '''

        self.name = name
        self.objects = Queue()
        self.create_func = create_func
        self.warm_stride_size = warm_stride_size
        self.total_objects_allocated = 0

    def warm(self, amount):
        '''
        Initializes the specified amount of objects.

        '''

        for i in range(amount):
            self.objects.put(self.create_func())

        self.total_objects_allocated += amount
    
    def get(self, verbose=True):
        '''
        Gets an object from the pool.

        :param verbose:
            Indicates whether this method should log warnings/errors. Defaults to ``True``.

        '''

        if self.objects.empty():
            if verbose:
                logging.warn('Exhausted object pool storage (currently allocated {} objects). Creating {} more.'.format(
                    self.total_objects_allocated, self.warm_stride_size
                ))

            self.warm(self.warm_stride_size)
        
        return self.objects.get()

    def free(self, object_to_free):
        '''
        Adds the specified object back to the pool.

        :param object_to_free:
            The object to add back to the pool.

        '''

        self.objects.put(object_to_free)

    def free_multiple(self, objects):
        '''
        Adds the specified objects back to the pool.

        :param objects:
            An array-like object containing the objects to add back to the pool.

        '''

        for _object in objects:
            self.free(_object)