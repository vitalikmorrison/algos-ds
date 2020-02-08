""" DECORATORS
    Nested Decorators
    Class decorators
    Note: Using functools.wraps retains original function's name and module
"""

from functools import wraps

def ex_decorator(func):
    """ Decorator pattern """
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator


@ex_decorator
def test_function(num):
    return num*num

print(test_function(5))


def try_several_times(no_attempts=5):
    """ Decorator with keyword arg """
    def decorator_try_func(func):
        @wraps(func)
        def wrap_try(*args, **kwargs):
            vals = []
            for i in range(no_attempts):
                value = func(*args, **kwargs)
                vals.append(value)
            return vals
        return wrap_try
    return decorator_try_func


@try_several_times(no_attempts=5)
def print_a_word(word):
    return word

print(print_a_word('now'))


def count_calls(func):
    @wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print(f"Call {wrapper_count_calls.num_calls} of {func.__name__!r}")
        return func(*args, **kwargs)
    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls


def cache(func):
    """ Keep a cache of previous function calls"""
    @wraps(func)
    def wrapper_cache(*args, **kwargs):

        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]

    wrapper_cache.cache = dict()
    return wrapper_cache

@cache
@count_calls
def fibonacci(num):
    if num < 2:
        return num
    return fibonacci(num - 1) + fibonacci(num - 2)

fibonacci(8)

""" JSON OBJECT VALIDATION """
from flask import Flask, request, abort
app = Flask(__name__)


def validate_json(*expected_args):
    def decorator_validate_json(func):
        @wraps(func)
        def wrapper_validate_json(*args, **kwargs):
            json_object = request.get_json()
            for expected_arg in expected_args:
                if expected_arg not in json_object:
                    abort(400)
            return func(*args, **kwargs)
        return wrapper_validate_json
    return decorator_validate_json


import errno
import os
import signal
import time


def timeout(seconds=1, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator


# @timeout(seconds=1, error_message='too late')
@timeout()
def test():
    time.sleep(4)
# test()


""" TIMEOUT CLASS IMPLEMENTATION
    params:
        - no_attempts
        - list of acceptable exceptions
        - total timeout
        - timeout per attempt
"""


class TryMe:
    def __init__(self, no_attempts=2, exc_list=[], tot_timeout=5, timeout_per_attempt=1):
        self.no_attempts = no_attempts
        self.exc_list = exc_list
        self.tot_timeout = tot_timeout
        self.timeout_per_attempt = timeout_per_attempt
        self.error_msg_tot = 'total timed out'
        self.error_msg = 'per attempt timed out'

    def _handle_timeout_tot(self, signum, frame):
        raise TimeoutError(self.error_msg_tot)

    def _handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_msg)

    def run_function(self, func, *args, **kwargs):
        signal.signal(signal.SIGALRM, self._handle_timeout_tot)
        signal.alarm(self.tot_timeout)

        for i in range(self.no_attempts):
            try:
                result = self.run_single_attempt(func, args, kwargs)
            except Exception as ex:
                if type(ex) in self.exc_list:
                    continue
                raise
            else:
                signal.alarm(signal.SIG_DFL)
            return result

    def run_single_attempt(self, func, args, kwargs):
        # signal.signal(signal.SIGALRM, self._handle_timeout)
        # signal.alarm(self.timeout_per_attempt)
        signal.signal(signal.SIGPROF, self._handle_timeout)
        signal.setitimer(signal.ITIMER_PROF, self.timeout_per_attempt)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            signal.setitimer(signal.ITIMER_PROF, signal.SIG_DFL)

def return_word(word):
    return word

def index_error_func():
    test_list = [1, 2, 3]
    return test_list[len(test_list)+1]

def sleep_func(time_to_sleep=5):
    time.sleep(time_to_sleep)
    return True

def divide(x, y):
    return x / y

def sleep_divide(x, y, time_to_sleep):
    time.sleep(time_to_sleep)
    return x / y

def hang_func():
    while True:
        x = 1 + 1

def run_tests_class():
    # Simple Example
    assert(TryMe().run_function(return_word, 'word')) == 'word'

    # Timeout
    assert(TryMe().run_function(sleep_func, 0.5)) is True

    # Simple Division
    assert(TryMe().run_function(divide, 2, 1)) == 2.0

    # Except an Error
    assert(TryMe(exc_list=[ZeroDivisionError]).run_function(divide, 2, 0)) is None

    # Pass Args
    assert(TryMe(no_attempts=5, tot_timeout=5, timeout_per_attempt=1).run_function(sleep_func, 0.9)) is True

    # Trigger Total Timeout
    try:
        TryMe(no_attempts=10, tot_timeout=3, exc_list=[ZeroDivisionError]).run_function(sleep_divide, 2, 0, 0.9)
    except TimeoutError:
        assert True
    else:
        assert False

    # Per Attempt Timeout
    try:
        TryMe(timeout_per_attempt=0.5).run_function(hang_func)
    except TimeoutError:
        assert True
    else:
        assert False

    # Unacceptable Exception
    try:
        TryMe().run_function(divide, '2', '1')
    except TypeError:
        assert True
    else:
        assert False

run_tests_class()


""" TIMEOUT DECORATOR IMPLEMENTATION """


def try_function(total_timeout=3, per_attempt_timeout=1, no_attempts=2, accepted_exceptions=[]):
    def decorator(func):
        def _handle_timeout_tot(signum, frame):
            raise TimeoutError('Timed Out')

        def _handle_timeout(signum, frame):
            raise TimeoutError('Attempt Timed Out')

        def _try_single_attempt(func, args, kwargs):
            signal.signal(signal.SIGPROF, _handle_timeout)
            signal.setitimer(signal.ITIMER_PROF, per_attempt_timeout)
            try:
                res = func(*args, **kwargs)
                return res
            finally:
                signal.setitimer(signal.ITIMER_PROF, signal.SIG_DFL)

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout_tot)
            signal.alarm(total_timeout)

            for i in range(no_attempts):
                try:
                    result = _try_single_attempt(func, args, kwargs)
                except Exception as ex:
                    if type(ex) in accepted_exceptions:
                        continue
                    raise
                else:
                    signal.alarm(signal.SIG_DFL)

                return result

        # Alternatively:
        # return wraps(func)(wrapper)
        return wrapper

    return decorator

print('Running Tests Class')
run_tests_class()


def run_tests_decorator():
    # Simple Example
    @try_function()
    def return_word(word):
        return word
    assert return_word('word') == 'word'

    # Timeout
    @try_function()
    def sleep_func(naptime):
        time.sleep(naptime)
        return True
    assert sleep_func(0.5) is True

    # Simple Division
    @try_function()
    def divide(x, y):
        return x / y
    assert divide(2, 1) == 2

    # Except an Error
    @try_function(accepted_exceptions=[ZeroDivisionError])
    def divide_ex(x, y):
        return x / y
    assert divide_ex(2, 0) is None

    # Pass Args
    # assert(TryMe(no_attempts=5, tot_timeout=5, timeout_per_attempt=1).run_function(sleep_func, 0.9)) is True
    @try_function(no_attempts=5, total_timeout=5, per_attempt_timeout=1)
    def sleep_args(sleep_time):
        time.sleep(sleep_time)
        return True

    assert sleep_args(0.9) is True

    # Trigger Total Timeout
    @try_function(no_attempts=10, total_timeout=3, accepted_exceptions=[ZeroDivisionError])
    def sleep_divide_tot(x, y, naptime):
        time.sleep(naptime)
        return x / y

    try:
        # TryMe(no_attempts=10, tot_timeout=3, exc_list=[ZeroDivisionError]).run_function(sleep_divide, 2, 0, 0.9)
        sleep_divide_tot(2, 0, 0.9)
    except TimeoutError:
        assert True
    else:
        assert False

    # Per Attempt Timeout
    @try_function(per_attempt_timeout=0.5)
    def hang_func():
        while True:
            pass

    try:
        hang_func()
    except TimeoutError:
        assert True
    else:
        assert False

    # Unacceptable Exception
    @try_function()
    def divide_ex2(x, y):
        return x/y

    try:
        divide_ex2("2", "1")
    except TypeError:
        assert True
    else:
        assert False

print('Running Tests Decorator')
run_tests_decorator()


""" EXCEPTIONS """
def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        print('division by zero!')
    else:
        print('result is ', result)
    finally:
        print('executing finally clause')


""" GENERATORS """
def get_series_a_funds():
    file_name = "techcrunch.csv"

    lines = (line.rstrip() for line in open(file_name))

    list_line = (s.split(",") for s in lines)
    cols = next(list_line)
    company_dicts = (dict(zip(cols, data)) for data in list_line)

    funding = (
        int(company_dict["raisedAmt"])
        for company_dict in company_dicts
        if company_dict["round"] == "a"
    )

    total_series_a = sum(funding)
    print(f"Total series A fundraising: ${total_series_a}")

get_series_a_funds()
