#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cma
from sklearn import tree
import numpy as np


import json
import subprocess
import os
import random
import math
from enum import Enum


def sign(number):
    """Will return 1 for positive,
    -1 for negative, and 0 for 0"""
    try:return number/abs(number)
    except ZeroDivisionError:return 0

class EchantillonNotDefinedException(Exception):
    pass

class Excalibur():
    """
        Excalibur is a powerful tool to model, simplify and reform a legislation.

        It takes as input a population with data (e.g., salaries/taxes/subsidies)

        It provides two functionalities:
            1) factorisation of existing legislation based on an economist's goals
            2) efficiency to write reforms and evaluate them instantly

        Population is given
    """

    class Color(Enum):
            red = 1
            green = 2
            blue = 3

    def __init__(self,  target_variable, taxable_variable, price_of_no_regression=0):
        self._target_variable = target_variable
        self._taxable_variable = taxable_variable
        self._max_cost = 0
        self._population = None
        self._price_of_no_regression = price_of_no_regression

    def filter_only_likely_population(self):
        """
            Removes unlikely elements in the population
            TODO: This should be done by the population generator

        :param raw_population:
        :return: raw_population without unlikely cases
        """
        new_raw_population = []
        for case in self._raw_population:
            if (int(case['0DA']) <= 1950 or ('0DB' in case and int(case['0DA'] <= 1950))) and 'F' in case and int(case['F']) > 0:
                pass
            else:
                new_raw_population.append(case)
        self._raw_population = new_raw_population

    def filter_only_no_revenu(self):
        """
            Removes people who have a salary from the population

        :param raw_population:
        :return: raw_population without null salary
        """
        new_raw_population = []
        for case in self._raw_population:
            if case.get('1AJ', 0) < 1 and case.get('1BJ', 0) < 1:
                new_raw_population.append(case)
        self._raw_population = new_raw_population

    def is_optimized_variable(self, var):
        return var != self._taxable_variable and var != self._target_variable

    def init_parameters(self, parameters, tax_rate_parameters=[], tax_threshold_parameters=[]):
        print repr(parameters)
        var_total = {}
        var_occurences = {}
        self._index_to_variable = []
        self._all_coefs = []
        self._var_to_index = {}
        self._var_tax_rate_to_index = {}
        self._var_tax_threshold_to_index = {}
        self._tax_rate_parameters = []
        self._tax_threshold_parameters = []
        self._parameters = set(parameters)
        index = 0
        for person in self._population:
            for var in parameters:
                if var in person:
                    if var not in self._var_to_index:
                        self._index_to_variable.append(var)
                        var_total[var] = 0
                        var_occurences[var] = 0
                        self._var_to_index[var] = index
                        index += 1
                    var_total[var] = var_total.get(var, 0) + person[var]
                    var_occurences[var] = var_occurences.get(var, 0) + 1

        for var in self._index_to_variable:
            self._all_coefs.append(var_total[var] / var_occurences[var])

        for var in tax_rate_parameters:
            self._all_coefs.append(0)
            self._var_tax_rate_to_index[var] = index
            self._tax_rate_parameters.append(var)
            index += 1

        for var in tax_threshold_parameters:
            self._all_coefs.append(5000)
            self._var_tax_threshold_to_index[var] = index
            self._tax_threshold_parameters.append(var)
            index += 1

    def find_all_possible_inputs(self, input_variable):
        possible_values = set()
        for person in self._population:
            if input_variable in person:
                if person[input_variable] not in possible_values:
                    possible_values.add(person[input_variable])
        return sorted(possible_values)

    def find_min_values(self, input_variable, output_variable):
        min_values = {}
        for person in self._population:
            if input_variable not in person:
                continue
            input = person[input_variable]
            if person[output_variable] <= min_values.get(input, 100000):
                min_values[input] = person[output_variable]
        return min_values

    def find_average_values(self, input_variable, output_variable):
        values = {}
        number_of_values = {}
        for person in self._population:
            if input_variable not in person:
                continue
            input = person[input_variable]
            values[input] = values.get(input, 0) + person[output_variable]
            number_of_values[input] = number_of_values.get(input, 0) + 1
        for input in values:
            values[input] = values[input] / number_of_values[input]
        return values

    def find_jumps_rec(self, init_jump_size, possible_inputs, values):
        if init_jump_size > 10000:
            return
        jumps = []
        for i in range(1, len(possible_inputs)):
            if abs(values[possible_inputs[i]] - values[possible_inputs[i-1]]) > init_jump_size:
                jumps.append(possible_inputs[i])

        if len(jumps) > 0 and len(jumps) < 5:
            return jumps
        else:
            return self.find_jumps_rec(init_jump_size * 1.1 , possible_inputs, values)

    def find_jumps(self, input_variable, output_variable, jumpsize=10, maxjumps=5, method='min'):
        """
            This function find jumps in the data

        """
        possible_inputs = self.find_all_possible_inputs(input_variable)

        # For binary values, jump detection is useless
        if len(possible_inputs) < 3:
            print 'No segmentation made on variable ' + input_variable + ' because it has less than 3 possible values'
            return []

        if method == 'min':
            values = self.find_min_values(input_variable, output_variable)
        elif method == 'average':
            values = self.find_average_values(input_variable, output_variable)
        else:
            assert False, 'Method to find the average value is badly defined, it should be "min" or "average"'

        jumps = self.find_jumps_rec(jumpsize, possible_inputs, values)

        if len(jumps) <= maxjumps:
            return jumps
        else:
            print 'No segmentation made on variable ' + input_variable + ' because it has more than ' \
                                                                       + str(maxjumps + 1) + ' segments'
            return []

    def add_segments_for_variable(self, variable):
        jumps = self.find_jumps(variable, self._target_variable)

        print 'Jumps for variable ' + variable + ' are ' + repr(jumps)

        if len(jumps) == 0:
            return []

        segment_names = []

        # First segment
        segment_name = variable + ' < ' + str(jumps[0])
        segment_names.append(segment_name)
        for person in self._population:
            if variable in person and person[variable] < jumps[0]:
                person[segment_name] = 1

        # middle segments
        for i in range(1, len(jumps)):
            if abs(jumps[i-1]-jumps[i]) > 1:
                segment_name = str(jumps[i-1]) + ' <= ' + variable + ' < ' + str(jumps[i])
            else:
                segment_name = variable + ' is ' + str(jumps[i-1])
            segment_names.append(segment_name)
            for person in self._population:
                if variable in person and person[variable] >= jumps[i-1] and person[variable] < jumps[i]:
                    person[segment_name] = 1

        # end segment
        segment_name = variable + ' >= ' + str(jumps[-1])
        segment_names.append(segment_name)
        for person in self._population:
            if variable in person and person[variable] >= jumps[-1]:
                person[segment_name] = 1

        return segment_names

    def add_segments(self, parameters, segmentation_parameters):
        new_parameters = []
        for variable in segmentation_parameters:
            new_parameters = new_parameters + self.add_segments_for_variable(variable)
        new_parameters = sorted(new_parameters)
        return parameters + new_parameters

    def simulated_target(self, person, coefs):
        simulated_target = 0
        threshold = 0
        tax_rate = 0
        for var in person:
            if var in self._parameters:
                idx = self._var_to_index[var]

                # Adding linear constant
                simulated_target += coefs[idx] * person[var]
            if var in self._tax_threshold_parameters:
                idx = self._var_tax_threshold_to_index[var]

                # determining the threshold from which we pay the tax
                threshold += coefs[idx] * person[var]
            if var in self._tax_rate_parameters:
                idx = self._var_tax_rate_to_index[var]

                # determining the tax_rate, divided by 100 to help the algorithm converge faster
                tax_rate += coefs[idx] * person[var] / 100

        simulated_target += person[self._taxable_variable] - (person[self._taxable_variable] - threshold / 10) * tax_rate
        return simulated_target

    def compute_cost_error(self, simulated, person):
        cost = simulated - person[self._target_variable]
        error = abs(cost)
        return cost, error

    def pissed_off_bricks(self, person, simulated):
        # TODO: clean this
        target_variation = (simulated - person[self._target_variable]) / (person[self._target_variable] + 0.01)
        if target_variation < 0:
            return -target_variation
        else:
            return 0

    def objective_function(self, coefs):
        error = 0
        error2 = 0
        total_cost = 0
        pissed_off_people = 0

        nb_people = len(self._population)

        for person in self._population:
            simulated = self.simulated_target(person, coefs)
            this_cost, this_error = self.compute_cost_error(simulated, person)
            total_cost += this_cost
            error += this_error
            error2 += error * error
            pissed_off_people += self.pissed_off_bricks(person, simulated)

        percentage_pissed_off = float(pissed_off_people) / float(nb_people)

        if random.random() > 0.99:
            print 'Best: avg change per month: ' + repr(int(error / (12 * len(self._population))))\
                  + ' cost: ' \
                  + repr(int(self.normalize_on_population(total_cost) / 1000000))\
                  + ' M/year and '\
                  + repr(int(1000 * percentage_pissed_off)/10) + '% people pissed ( -5% salary )'

        cost_of_overbudget = 100000
        cost_of_pissed_of_people = 10000000

        if self.normalize_on_population(total_cost) > self._max_cost:
            error2 += pow(cost_of_overbudget, 2) * self.normalize_on_population(total_cost)

        if -self.normalize_on_population(total_cost) < self._min_saving:
            error2 += pow(cost_of_overbudget, 2) * self.normalize_on_population(total_cost)

        if percentage_pissed_off > (1 - self._percent_not_pissed_off):
            error2 += pissed_off_people * pow(cost_of_pissed_of_people, 2)

        return math.sqrt(error2)

    def find_useful_parameters(self, results, threshold=100):
        """
            Eliminate useless parameters
        """
        new_parameters = []
        optimal_values = []
        for i in range(len(results)):
            if results[i] >= threshold:
                new_parameters.append(self._index_to_variable[i])
                optimal_values.append(results[i])
            else:
                print 'Parameter ' + self._index_to_variable[i] + ' was dropped because it accounts to less than '\
                      + str(threshold) + ' euros'
        return new_parameters, optimal_values

    def suggest_reform(self, parameters, max_cost=0, min_saving=0, verbose=False, tax_rate_parameters=[], tax_threshold_parameters=[], percent_not_pissed_off=0):
        """
            Find parameters of a reform

        :param parameters: variables that will be taken into account
        :param max_cost: maximum cost of the reform in total, can be negative if we want savings
        :param verbose:
        :return: The reform for every element of the population
        """

        self._percent_not_pissed_off = percent_not_pissed_off
        self._max_cost = max_cost
        self._min_saving = min_saving

        if (self._max_cost != 0 or self._min_saving != 0) and self._echantillon is None:
            raise EchantillonNotDefinedException()

        if verbose:
            cma.CMAOptions('verb')

        self.init_parameters(parameters,
                             tax_rate_parameters=tax_rate_parameters,
                             tax_threshold_parameters=tax_threshold_parameters)

        # new_parameters = self.add_segments(direct_parameters,  barem_parameters)
        # self.init_parameters(new_parameters)

        res = cma.fmin(self.objective_function, self._all_coefs, 10000.0, options={'maxfevals': 5e3})

        # print '\n\n\n Reform proposed: \n'
        #
        final_parameters = []

        i = 0
        while i < len(self._index_to_variable):
            final_parameters.append({'variable': self._index_to_variable[i],
                                     'value': res[0][i],
                                     'type': 'base_revenu'})
            i += 1

        offset = len(self._index_to_variable)

        while i < offset + len(self._tax_rate_parameters):
            final_parameters.append({'variable': self._tax_rate_parameters[i-offset],
                                     'value': res[0][i],
                                     'type': 'tax_rate'})
            i += 1

        offset = len(self._index_to_variable) + len(self._tax_rate_parameters)
        while i < offset + len(self._tax_threshold_parameters):
            final_parameters.append({'variable': self._tax_threshold_parameters[i-offset],
                                     'value': res[0][i],
                                     'type': 'tax_threshold'})
            i += 1

        simulated_results, error, cost = self.apply_reform_on_population(self._population, coefficients=res[0])
        return simulated_results, error, cost, final_parameters


    def population_to_input_vector(self, population):
        output = []
        for person in population:
            person_output = self.person_to_input_vector(person)
            output.append(person_output)
        return output

    def person_to_input_vector(self, person):
        return list(person.get(var, 0) for var in self._index_to_variable)

    def suggest_reform_tree(self,
                            parameters,
                            max_cost=0,
                            min_saving=0,
                            verbose=False,
                            max_depth=3,
                            image_file=None,
                            min_samples_leaf=2):
        self._max_cost = max_cost
        self._min_saving = min_saving

        if (self._max_cost != 0 or self._min_saving != 0) and self._echantillon is None:
            raise EchantillonNotDefinedException()

        self.init_parameters(parameters)

        X = self.population_to_input_vector(self._population)
        y = map(lambda x: int(x[self._target_variable]), self._population)

        clf = tree.DecisionTreeRegressor(max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf)
        clf = clf.fit(X, y)

        simulated_results, error, cost = self.apply_reform_on_population(self._population, decision_tree=clf)

        if image_file is not None:
            with open( image_file + ".dot", 'w') as f:
                f = tree.export_graphviz(clf,
                                         out_file=f,
                                         feature_names=self._index_to_variable,
                                         filled=True,
                                         impurity=True,
                                         proportion=True,
                                         rounded=True,
                                         rotate=True
                                         )
            os.system('dot -Tpng ' + image_file + '.dot -o ' + image_file + '.png')
            # 'dot -Tpng enfants_age.dot -o enfants_age.png')
                             # ')

        # dot_data = tree.export_graphviz(clf)
        #
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("new_law.pdf")
        #
        # dot_data = tree.export_graphviz(clf, out_file=None,
        #                          feature_names=self._index_to_variable,
        #                          filled=True, rounded=True,
        #                          special_characters=True)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # Image(graph.create_png())
        return simulated_results, error, cost, clf

    def is_boolean(self, variable):
        """
            Defines if a variable only has boolean values

        :param variable: The name of the variable of interest
        :return: True if all values are 0 or 1, False otherwise
        """
        for person in self._population:
            if variable in person and person[variable] not in [0, 1]:
                return False
        return True

    def apply_reform_on_population(self, population, coefficients=None, decision_tree=None):
        """
            Computes the reform for all the population

        :param population:
        :param coefficients:
        :return:
        """
        simulated_results = []
        total_error = 0
        total_cost = 0
        for i in range(0, len(population)):
            if decision_tree:
                simulated_result = float(decision_tree.predict(self.person_to_input_vector(population[i]))[0])
            elif coefficients is not None:
                simulated_result = self.simulated_target(population[i], coefficients)
            simulated_results.append(simulated_result)
            this_cost, this_error = self.compute_cost_error(simulated_result, population[i])
            total_cost += this_cost
            total_error += this_error

        total_cost = self.normalize_on_population(total_cost)

        return simulated_results, total_error / len(population), total_cost

    def add_concept(self, concept, function):
        if self._population is None:
            self._population = list(map(lambda x: {self._target_variable: x[self._target_variable]}, self._raw_population))

        for i in range(len(self._raw_population)):
            result = function(self._raw_population[i])
            if result is not None and result is not False:
                self._population[i][concept] = float(result)

    def normalize_on_population(self, cost):
        if self._echantillon is None or self._echantillon == 0:
            raise EchantillonNotDefinedException()
        return cost / self._echantillon

    def summarize_population(self):
        total_people = 0
        for family in self._raw_population:
            total_people += 1
            if '0DB' in family and family['0DB'] == 1:
                total_people += 1
            if 'F' in family:
                total_people += family['F']

        # We assume that there are 2000000 people with RSA
        # TODO Put that where it belongs in the constructor
        self._echantillon =  float(total_people) / 2000000
        print 'Echantillon of ' + repr(total_people) + ' people, in percent of french population for similar revenu: ' + repr(100 * self._echantillon) + '%'


    def load_from_json(self, filename):
        with open('../results/' + filename, 'r') as f:
            return json.load(f)

    def load_openfisca_results(self, filename):
        results_openfisca = self.load_from_json(filename + '-openfisca.json')
        testcases = self.load_from_json(filename + '-testcases.json')
        self._raw_population = testcases[:]
        for i in range(len(testcases)):
            self._raw_population[i][self._target_variable] = results_openfisca[i][self._target_variable]