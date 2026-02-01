# settings.py


def init(args_instance=None):
    global runargs
    global linelist
    global cia
    global cloudata
    global samples
    # global saverunargs
    samples = []
    if args_instance is not None:
        runargs = args_instance
    else:
        runargs = 0
        
