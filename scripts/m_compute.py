"""
Compare with http://www3.finances.gouv.fr/calcul_impot/2015/index.htm

"""


import json
import math


def product(l):
    accu = 1.
    for e in l:
        accu *= e
    return accu


def boolean_or(l):
    for e in l:
        if e:
            return 1.
    return 0.


def boolean_et(l):
    for e in l:
        if not l:
            return 0.
    return 1.


functions_mapping = {
    'sum': sum,
    '+': sum,
    'product': product,
    'negate': (lambda x: -x[0]),
    'unary:-': (lambda x: -x[0]),
    'positif': (lambda x: float(x[0]>0)),
    'positif_ou_nul': (lambda x: float(x[0]>=0)),
    'null': (lambda x: float(x[0]==0)),
    'operator:>=': (lambda x: float(x[0]>=x[1])),
    'operator:<=': (lambda x: float(x[0]<=x[1])),
    'operator:>': (lambda x: float(x[0]>x[1])),
    'operator:<': (lambda x: float(x[0]<x[1])),
    'operator:=': (lambda x: float(x[0]==x[1])),
    'ternary': (lambda x: x[1] if x[0] else x[2]),
    'si': (lambda x: x[1] if x[0] else 0.),
    'invert': (lambda x: 1/x[0] if x[0] else 0.),
    'inverse': (lambda x: 1/x[0] if x[0] else 0.),
    'max': max,
    'min': min,
    'inf': (lambda x: float(math.floor(x[0]))),
    'arr': (lambda x: float(round(x[0]))),
    'abs': (lambda x: abs(x[0])),
    'present': (lambda x: float(x[0] != 0.)),
    'boolean:ou': boolean_or,
    'boolean:et': boolean_et,
    'dans': (lambda x: 1. if (x[0] in x[1:]) else 0.)
}

with open('../light_ast/computing_order.json', 'r') as f:
    computing_order = json.load(f)

with open('../light_ast/children_light.json', 'r') as f:
    children_light = json.load(f)

with open('../light_ast/formulas_light.json', 'r') as f:
    formulas_light = json.load(f)

with open('../light_ast/constants_light.json', 'r') as f:
    constants_light = json.load(f)

with open('../light_ast/inputs_light.json', 'r') as f:
    inputs_light = json.load(f)

with open('../light_ast/unknowns_light.json', 'r') as f:
    unknowns_light = json.load(f)

with open('../light_ast/input_variables.json', 'r') as f:
    input_variables = json.load(f)

alias2name = {i['alias']: i['name'] for i in input_variables}

def get_value(name, input_values, computed_values):
    if name in formulas_light:
        return computed_values[name]

    if name in constants_light:
        return constants_light[name]

    if name in inputs_light:
        return input_values[name]

    if name in unknowns_light:
        return 0.

    raise Exception('Unknown variable category.')


def prepare(alias_values):
    alias2name['year'] = 'year'
    input_values = {alias2name[alias]: value for alias, value in alias_values.items()}

    input_values_complete = {}
    for name in inputs_light:
        if (name in input_values):
            input_values_complete[name] = input_values[name]
        else:
            input_values_complete[name] = 0.

    return input_values_complete


def compute_formula(node, input_values, computed_values, debug_variable=None, depth=0):
    nodetype = node['nodetype']

    if nodetype == 'symbol':
        name = node['name']
        value = get_value(name, input_values, computed_values)
        if debug_variable:
            indent = '   ' * depth
            print '|' + indent + '- symbol ' + name + ' value ' + str(value)
        return value

    if nodetype == 'float':
        value = node['value']
        if debug_variable:
            indent = '   ' * depth
            print '|' + indent + '- ' + str(value)
        return value

    if nodetype == 'call':
        name = node['name']
        if debug_variable:
            indent = '   ' * depth
            print '|' + indent + '- call ' + name + ' - ' + str(len(node['args']))
        args = [compute_formula(child, input_values, computed_values, debug_variable, depth + 1) for child in node['args']]
        function = functions_mapping[name]
        value = function(args)
        if debug_variable:
            indent = '   ' * depth
            print '|' + indent + '-= ' + str(value)
        return value

    raise ValueError('Unknown type : %s'%nodetype)


def compute(input_values):
    computed_values = {}

    for variable in computing_order:
        formula = formulas_light[variable]
        computed_values[variable] = compute_formula(formula, input_values, computed_values)

    important_vars = ['NBPT', 'REVKIRE', 'BCSG', 'BCSG', 'BRDS', 'IBM23', 'TXMOYIMP', 'NAPTIR', 'IINET', 'RRRBG', 'RNI', 'IDRS3', 'IAVIM', 'IRN']
    return {var: computed_values[var] for var in formulas_light}


def m_compute_from_aliases(alias_values):
    input_values = prepare(alias_values)
    return compute(input_values)


