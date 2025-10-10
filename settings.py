# settings.py

# def init():
#     global runargs
#     global linelist
#     global samples
#     samples = []
#     runargs = 0


def init(args_instance=None):
    global runargs
    global linelist
    global cia
    global cloudata
    # global saverunargs
    samples = []
    if args_instance is not None:
        runargs = args_instance
    else:
        runargs = 0
        
