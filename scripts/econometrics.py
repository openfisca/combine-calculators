import matplotlib.pyplot as plt

def get_cumulative(revs):
    revs = sorted(revs)
    cum_revs = []
    current = 0
    for rev in revs:
        current = current + rev
        cum_revs.append(current)
    return cum_revs

def draw_ginis(old_revenus, new_revenus):
    fig = plt.figure()
    ax = fig.add_axes((0.5, 0.5, 1, 1))
    ax.set_xlabel('Cummulative population')
    ax.set_ylabel('Cummulative revenu')

    n = len(old_revenus)

    cum_new_revenus = get_cumulative(new_revenus)
    cum_old_revenus = get_cumulative(old_revenus)

    ax.plot(
        range(n),
        cum_old_revenus,
        alpha=1,
        label='Old Revenus',
        c='blue')
    ax.plot(
        range(n),
        cum_new_revenus,
        alpha=1,
        label='New Revenus',
        c='red')

    ax.legend(loc='upper right')
    plt.title('Gini before and after reform:')



def gini(revdisp):
    """Gini computed according to
       https://en.wikipedia.org/wiki/Gini_coefficient

       Arg:
            revdisp: an iterable of available revenu for the population

       Return:
            gini: value between 0 and 1

    """
    revdisp = sorted(revdisp)


    sum_y = 0
    sum_yi = 0
    n = len(revdisp)

    i = 1
    for rev in revdisp:
        sum_y += rev
        sum_yi += i * rev
        i += 1

    return 2 * sum_yi / (n * sum_y) - (n + 1) / n
