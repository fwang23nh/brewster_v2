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

    linelist = args_instance.linelist
    cia = args_instance.cia
    if hasattr(args_instance, "cloudata"):
        cloudata = args_instance.cloudata
