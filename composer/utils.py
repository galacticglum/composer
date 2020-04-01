'''
Utility methods.

'''

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
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
    :returns:
        A list of the form [function(array[0]), function(array[1]), ...].

    '''
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]

    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]

    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]

        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }

        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs): pass

    out = []
    
    # Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)

    return front + out