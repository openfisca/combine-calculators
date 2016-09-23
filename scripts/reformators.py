#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cma
import math
import random

FRENCH_POPULATION = 61000000

class Excalibur():
    """
        Excalibur is a powerful tool to model, simplify and reform a legislation.

        It takes as input a population with data (e.g., salaries/taxes/subsidies) and/or a simulator (e.g., openfisca)

        It provides two functionalities:
            1) factorisation of existing legislation based on an economist's goals
            2) efficiency to write reforms and evaluate them instantly

        Population is given
    """

    def __init__(self, population, target_variable, taxable_variable, simulator=None, echantillon=None):
        self._population = population[:]
        self._simulator = simulator
        self._target = target_variable
        self._taxable_variable = taxable_variable
        self._save_per_person = 0
        self._echantillon = echantillon

    def is_optimized_variable(self, var):
        return var != self._taxable_variable and var != self._target

    def init_parameters(self, parameters):
        print repr(parameters)
        var_total = {}
        var_occurences = {}
        self._index_to_variable = []
        self._all_coefs = []
        self._var_to_index = {}
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

        jumps = []
        for i in range(1, len(possible_inputs)):
            if abs(values[possible_inputs[i]] - values[possible_inputs[i-1]]) > jumpsize:
                jumps.append(possible_inputs[i])

        if len(jumps) <= maxjumps:
            return jumps
        else:
            print 'No segmentation made on variable ' + input_variable + ' because it has more than ' \
                                                                       + str(maxjumps + 1) + ' segments'
            return []

    def add_segments_for_variable(self, variable):
        jumps = self.find_jumps(variable, self._target)

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
        for var in person:
            if var in self._parameters:
                idx = self._var_to_index[var]
                # Adding linear constant
                simulated_target += coefs[idx] * person[var]
        return simulated_target

    def objective_function(self, coefs):
        error = 0
        error2 = 0
        total_saving = 0

        for person in self._population:
            this_saving = person[self._target] - self.simulated_target(person, coefs)
            total_saving += this_saving
            error += abs(this_saving)
            error += this_saving * this_saving

        if total_saving / self._echantillon < self._save:
            error *= 2

        if random.random() > 0.99:
            print 'Average error per person = ' + repr(int(error/len(self._population))) + ' saving '\
                  + repr(int(total_saving/(self._echantillon * 1000000))) + ' millions'
        return error * error

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

    def suggest_reform(self, boolean_parameters, linear_parameters=None, barem_parameters=None, save=0,
                       verbose=False):
        """
            Find parameters of a reform

        :param boolean_parameters: have 0 or 1 value (parameter not present is considered 0)
        :param linear_parameters: the result will be proportional to these parameters
        :param segmentation_parameters: these parameters will be segmented into different cases
        :param save: how much money we would like to save in total
        :param verbose:
        :return: The reform for every element of the population
        """
        if not linear_parameters:
            linear_parameters = []
        if not barem_parameters:
            barem_parameters = []

        direct_parameters = boolean_parameters + linear_parameters

        self._save = save

        if save != 0 and self._echantillon == None:
            print 'You want to decrease spending but you did not define echantillon. We cannot do that.'
            print 'For instance, if your population is made of unemployed people that represent 1% of the total' \
                  'unemployed people, you should define echantillon = 0.01 in the constructor of Excalibur.'
            assert self._echantillon != None

        print 'Population size = ' + repr(len(self._population))

        if verbose:
            cma.CMAOptions('verb')

        new_parameters = self.add_segments(direct_parameters,  barem_parameters)
        self.init_parameters(new_parameters)

        res = cma.fmin(self.objective_function, self._all_coefs, 10000.0, options={'maxfevals': 3e3})

        print '\n\n\n Reform proposed: \n'

        for i in range(0, len(self._index_to_variable)):
            if self.is_boolean(self._index_to_variable[i]):
                print '+ ' + str(int(res[0][i])) + '€ \t if ' + self._index_to_variable[i]
            else:
                print '+ ' + str(int(res[0][i])) + '€ x ' + self._index_to_variable[i]

        return self.apply_reform_on_population(self._population, res[0])

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

    def apply_reform_on_population(self, population, coefficients):
        """
            Computes the reform for all the population

        :param population:
        :param coefficients:
        :return:
        """
        simulated_results = []
        for i in range(0, len(population)):
            simulated_results.append(self.simulated_target(population[i], coefficients))
        return simulated_results