# -*- coding: UTF-8 -*-

"""
    Simulate a population of N inhabitants from the statistics on the tax declaration.

    Statistiques utilisé pour la période 2014
"""

import random


# STATISTICS

# Possible situations:
POSSIBLE_SITUATIONS = [('M', 12002841), ('D', 5510809), ('O', 983754), ('C', 14934477), ('V', 3997578)]

# Statistiques by familly, approximated to match the declaration d'impots
CHILDREN_PER_FAMILY = [(1, 46), (2, 38.5), (3, 12.5), (4, 2), (5, 1)]

# Familles avec enfants a charge de moins de 18 ans
FAMILLES = 9321480

# Declarations:
TOTAL_DECLARATIONS = sum(w for c, w in POSSIBLE_SITUATIONS)

REVENUS = [21820704, 503723963299]
REVENUS_AVG = REVENUS[1] / float(REVENUS[0])
REVENUS_0 = REVENUS[0] / float(TOTAL_DECLARATIONS)

def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

def generate_random_cerfa():
    cerfa = {}

    # Age is random. Very bad approximation. But why not.
    cerfa['0DA'] = int(random.random() * 70 + 18)

    # Drawing the situation
    situation = weighted_choice(POSSIBLE_SITUATIONS)
    cerfa[situation] = 1

    ## We only give children to married or pacces. This is an approximation
    enfants = 0
    if situation == 'M' or situation == 'O':
        if random.random() < (FAMILLES / float(POSSIBLE_SITUATIONS[0][1] + POSSIBLE_SITUATIONS[2][1])):
            enfants = weighted_choice(CHILDREN_PER_FAMILY)

    if enfants > 0:
        cerfa['F'] = enfants

    # Distribution that has a cool shape and required properties
    cerfa['1AJ'] = max(random.gauss(5500, 26500), 0)
    return cerfa['1AJ']