import cma
import math
import random

class Excalibur():
    """
        Excalibur is a powerful tool to model, simplify and reform a legislation.

        It takes as input a population with data (e.g., salaries/taxes/subsidies) and/or a simulator (e.g., openfisca)

        It provides two functionalities:
            1) factorisation of existing legislation based on an economist's goals
            2) efficiency to write reforms and evaluate them instantly

        Population is given
    """

    def __init__(self, population, target_variable, taxable_variable, simulator=None):
        self._population = population
        self._simulator = simulator
        self._target = target_variable
        self._taxable_variable = taxable_variable

    def is_optimized_variable(self, var):
        return var != self._taxable_variable and var != self._target

    def init_parameters(self, parameters):
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
        quad_error = 0
        for person in self._population:
            this_error = abs(self.simulated_target(person, coefs) - person[self._target])
            error += abs(this_error)
            quad_error += this_error * this_error
        if random.random() > 0.98:
            print 'Average error per person = ' + repr(int(error/len(self._population)))
        return error

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
                print 'Parameter ' + self._index_to_variable[i] + ' was dropped because it accounts to less than ' + str(threshold) + ' euros'
        return new_parameters, optimal_values

    def suggest_reform(self, parameters):
        print 'Population size= ' + repr(len(self._population))
        cma.CMAOptions('verb')

        self.init_parameters(parameters)
        res = cma.fmin(self.objective_function, self._all_coefs, 10000.0, options={'maxfevals': 3e3})

        print '\n\n\n Modeling as the sum of: \n'
        for i in range(0, len(self._index_to_variable)):
            print self._index_to_variable[i] + ' x ' + str(int(res[0][i]))

        # print ' plus % revenu imposable = '
        # for i in range(len(self._index_to_variable), 2 * len(self._index_to_variable)):
        #     print ' - ' + self._index_to_variable[i - len(self._index_to_variable)] + ' * ' + str(res[0][i]/10000) + ' * salaire_imposable'
        return res

    def compare_population_to_results(self, population, results):
        simulated_results = []
        for i in range(0, len(population)):
            simulated_results.append(self.simulated_target(population[i], results))
        return simulated_results