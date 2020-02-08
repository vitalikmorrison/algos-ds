# follow.py
#
# A generator that follows a log file like Unix 'tail -f'.
#
# Note: To see this example work, you need to apply to
# an active server log file.  Run the program "logsim.py"
# in the background to simulate such a file.  This program
# will write entries to a file "access-log".

import time
def follow(thefile):

    thefile.seek(0, 2)      # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)    # Sleep briefly
            continue
        yield line

"""
# Example use
if __name__ == '__main__':
    logfile = open("access-log")
    for line in follow(logfile):
        print(line)
"""

# A decorator function that takes care of starting a coroutine
# automatically on call.
import functools


def coroutine(func):
    @functools.wraps(func)
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


@coroutine
def grep(pattern):
    print(f"Looking for {pattern}")
    try:
        while True:
            line = (yield)
            if pattern in line:
                print(line,)
    except GeneratorExit:
        print("closing the shop - goodbye")

g = grep("python")
# Notice how you don't need a next() call here
g.send("Yeah, but no, but yeah, but no")
g.send("A series of tubes")
g.send("python generators rock!")
# g.throw(IndentationError)
g.close()


"""
cofollow.py

A simple example showing how to hook up a pipeline with
coroutines.   To run this, you will need a log file.
Run the program logsim.py in the background to get a data
source.


A data source.  This is not a coroutine, but it sends
data into one (target)
"""


import signal
from functools import wraps
def try_with_timeout(timeout=2):
    def decorator(func):

        def _handle_timeout(signum, frame):
            raise TimeoutError('Attempt Timed Out')

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(timeout)
            try:
                return func(*args, **kwargs)

            finally:
                signal.alarm(signal.SIG_DFL)

        return wrapper

    return decorator


@try_with_timeout()
def follow(thefile, target):
    thefile.seek(0, 2)      # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)    # Sleep briefly
            continue
        target.send(line)

# A sink.  A coroutine that receives data

@coroutine
def printer():
    while True:
        line = (yield)
        print(line)


f = open("access-log")
try:
    follow(f, printer())
except TimeoutError:
    print('func timed out ... moving on')


@try_with_timeout()
def follow(thefile, target):
    """ copipe.py """
    thefile.seek(0, 2)      # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)    # Sleep briefly
            continue
        target.send(line)

# Filter
@coroutine
def grep(pattern, target):
    while True:
        line = (yield)           # Receive a line
        if pattern in line:
            target.send(line)    # Send to next stage

# Sink - coroutine that receives data
@coroutine
def printer():
    while True:
        line = (yield)
        print(line,)

# Example
f = open("access-log")
try:
    follow( f, grep('python', printer()) )
except TimeoutError:
    print('func timed out ... moving on')
