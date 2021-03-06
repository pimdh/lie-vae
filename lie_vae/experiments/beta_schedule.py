from lie_vae.experiments.utils import ConstantSchedule, LinearSchedule


def get_beta_schedule(schedule, beta):
    if schedule is None and beta is not None:
        return ConstantSchedule(beta)
    elif schedule == 'a':
        return LinearSchedule(0.001, 1, 60000, 200000)
    elif schedule == 'b':
        return LinearSchedule(0.001, 0.1, 60000, 200000)
    elif schedule == 'c':
        return LinearSchedule(0.001, 0.01, 60000, 200000)
    elif schedule == 'd':
        return LinearSchedule(0.001, 10, 60000, 200000)
    elif schedule == 'e':
        return LinearSchedule(0.001, 0.1, 60000, 120000)
    elif schedule == 'f':
        return LinearSchedule(0.001, 1, 60000, 120000)
    elif schedule == 'g':
        return LinearSchedule(0.001, 0.3, 60000, 120000)
    elif schedule == 'h':
        return LinearSchedule(0.001, 0.3, 30000, 60000)
    elif schedule == 'i':
        return LinearSchedule(0.001, 1, 30000, 60000)
    elif schedule == 'j':
        return LinearSchedule(0.001, 3, 30000, 60000)
    elif schedule == 'k':
        return LinearSchedule(0.001, 10, 30000, 60000)
    elif schedule == 'l':
        return LinearSchedule(0.001, 30, 30000, 60000)
    elif schedule == 'm':
        return LinearSchedule(0.001, 3, 60000, 120000)
    elif schedule == 'n':
        return LinearSchedule(0.001, 10, 60000, 120000)
    elif schedule == 'o':
        return LinearSchedule(0.001, 30, 60000, 120000)
    elif schedule == 'p':
        return LinearSchedule(0.001, 100, 60000, 120000)
    elif schedule == 'q':
        return LinearSchedule(0.001, 10, 60000, 240000)
    elif schedule == 'r':
        return LinearSchedule(0.001, 30, 60000, 600000)
    elif schedule == 's':
        return LinearSchedule(0.001, 3, 60000, 240000)
    elif schedule == 'u':
        return LinearSchedule(3, 30, 60000, 120000)
    elif schedule == 'v':
        return LinearSchedule(3, 30, 30000, 60000)
    elif schedule == 'w':
        return LinearSchedule(1E-2, 1, 20000, 40000)
    elif schedule == 'x':
        return LinearSchedule(7, 15, 250000, 750000)
    elif schedule == 'y':
        return LinearSchedule(7, 20, 250000, 750000)
    elif schedule == 'z':
        return LinearSchedule(7, 30, 250000, 750000)
    else:
        raise RuntimeError('Wrong beta schedule. Schedule={}, beta={}'
                           .format(schedule, beta))
