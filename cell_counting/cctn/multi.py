import multiprocessing
import time

data = (
    ['a', '1'], ['b', '2'], ['c', '3'], ['c', '4'],
    ['e', '5'], ['f', '6'], ['g', '7'], ['h', '8']
)



def mp_worker(in_arg):
    letter, wait_time = in_arg
    print(nice)
    print("Process\t %s \t data: %s, waiting %s seconds." % (multiprocessing.current_process().name, letter, wait_time))
    time.sleep(int(wait_time))
    print("Process \t %s \t data: %s \t finished!!!" % (multiprocessing.current_process().name,letter))
    result=dict()
    result.update({wait_time : letter})
    return result


def mp_handler():
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    results = p.map(mp_worker, data)
    my_dict = dict()
    for curResult in results:
        my_dict.update(curResult)
    print(my_dict)


if __name__ == '__main__':
    times = range(1, 4)
    nice="How is this possible?"
    print(data)
    names = ['Name']*3
    print(tuple(zip(names, list(times))))
    # data = tuple(zip(names, list(times)))
    mp_handler()
