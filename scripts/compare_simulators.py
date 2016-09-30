# -*- coding: UTF-8 -*-

"""
    This script compares test results from various tax or benefit simulators

    The goals are:

    A) to evaluate the percentage of similarity between two simulators (e.g., M, http://www3.finances.gouv.fr/, openfisca)
    B) to enable to debug precisely which variables are faulty in case of discreapency in the results
    C) to match variables from one simulator to their equivalent on the other simulator
    D) to match the parameters of the law from M to openfisca

    It takes as input a list of tests done on two simulators with the same input parameters.
    For each test, the simulator outputs the results of its variables that it wants to automatically match
"""

from lxml import etree
from input_variable_converter import CerfaOpenFiscaConverter
from input_variable_converter import CerfaMConverter
from input_variable_converter import CerfaOnlineComparator

import openfisca_france, random, operator, time, requests
import argparse
import sys
import json

from m_compute import m_compute
from population_simulator import CerfaPopulationSimulator

"""
    The margin of error, which can be positive to allow for rounding errors
"""
ERROR_MARGIN = 0.001
NB_TESTS = 20
PRINTING_THRESHOLD = 0.9
PRINTING_MAX_LINES = 30

"""
    Example results of 3 simulations
"""

class TaxbenefitSimulator():
    def __init__(self, testcases):
        self._testcases = testcases

    def simulate(self, verbose=False):
        results = []
        result_count = 0
        for test in self._testcases:
            result = None
            while result is None:
                try:
                    result = self.simulate_one_test(test)
                except ArithmeticError as e:
                    print repr(e)
                    time.sleep(1)
                    print 'Error simulating, trying again...'
            results.append(result)
            result_count += 1
            if verbose:
                print ('{} simulations done'.format(result_count))
        return results

    def simulate_one_test(self, test):
        raise NotImplementedError

class OpenfiscaSimulator(TaxbenefitSimulator):
    def __init__(self, testcases):
        self._tax_benefit_system = openfisca_france.FranceTaxBenefitSystem()
        self._testcases = testcases
        self._converter = CerfaOpenFiscaConverter()

    def find_column_by_alias(self, alias):
        columns = self._tax_benefit_system.column_by_name.values()
        matches = list(filter(lambda column: column.cerfa_field is not None and alias in column.cerfa_field, columns))
        assert len(matches) in (0, 1), (alias, matches)
        return matches[0] if matches else None


    def find_cerfa_conversions(self):
        for alias in simple_input_variables_with_range:
            column = self.find_column_by_alias(alias)
            if column is not None:
                print (alias, column.name)
            else:
                print ('f' + alias.lower(), ' None ',
                       self._tax_benefit_system.column_by_name.get('f' + alias.lower()))

    def make_scenario(self, v):
        openfisca_inputs = self._converter.convert_cerfa_to_openfisca(v, v['year'])
        print repr(openfisca_inputs)
        return self._tax_benefit_system.new_scenario().init_single_entity(
            **openfisca_inputs)

    def simulate_one_test(self, test):
        scenario = self.make_scenario(test)
        simulation = scenario.new_simulation(trace=True)

        revdisp = float(simulation.calculate('revdisp', '2014', print_trace=False, max_depth=10, show_default_values=False)[0])
        irpp = float(-simulation.calculate('irpp', '2014')[0])
        credits_impot = float(simulation.calculate('credits_impot', '2014')[0])
        salaire_imposable = float(simulation.calculate('salaire_imposable', '2014')[0])
        taux_moyen_imposition = float(simulation.calculate('taux_moyen_imposition', '2014')[0]) * 100
        tot_impot = float(simulation.calculate('tot_impot', '2014')[0])
        print 'Irpp openfisca = ' + repr(irpp)
        print 'Revdisp openfisca = ' + repr(revdisp)
        return {'irpp': irpp, 'credits_impot': credits_impot, 'salaire_imposable': salaire_imposable,
                'taux_moyen_imposition': taux_moyen_imposition, 'tot_impot': tot_impot, 'revdisp': revdisp}

class MDescriptions():
    """
        For each variable, gets the name of the variable in the file TGVH.m
        TODO: which link to tgvh to use?
    """
    def __init__(self):
        self._variable_to_descriptions = {}
        # TODO: auto update m source
        with open('../m/tgvH.m', 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Corrects double space in the m file
                newline = line.replace('  :', ' :')
                elements = newline.split(' : ')
                if len(elements) > 1:
                    self._variable_to_descriptions[elements[0]] = elements[0] + ' (' + elements[-1].strip().replace('"',
                                                                                                                    '').replace(
                        ';', '') + ')'

    def get_description(self, variable):
        return self._variable_to_descriptions[variable]

class MSimulator(TaxbenefitSimulator):
    def __init__(self, testcases, m_descriptions):
        self._testcases = testcases
        self._m_descriptions = m_descriptions
        self._converter = CerfaMConverter()

    def simulate_one_test(self, test):
        """
            Calls the M code and translate the name of the variable to a human readable name
        """
        m_inputs = self._converter.convert_cerfa_to_m(test)
        not_null_inputs = filter(lambda inpu: m_inputs[inpu] > 0, m_inputs)

        print 'M inputs are' + repr(list([ str(input) + ' ' + str(m_inputs[input]) for input in not_null_inputs]))
        result = m_compute(m_inputs)
        print 'IRN= ' + repr(int(result['IRN']))
        return {variable: result[variable] for variable in result}

class OnlineTaxSimulator(TaxbenefitSimulator):
    def __init__(self, testcases, m_descriptions):
        self._testcases = testcases
        self._m_descriptions = m_descriptions
        self._comparator = CerfaOnlineComparator()

    def simulate_one_test(self, test):
        saisie_variables = self._comparator.convert_cerfa_to_online(test)
        # Year is the year of the tax period
        print 'Inputs online = ' + repr(saisie_variables)
        year_calc =  int(test.get('year', 2014)) + 1
        cgi_url = 'http://www3.finances.gouv.fr/cgi-bin/calc-{}.cgi'.format(year_calc)
        headers = {'User-Agent': 'Calculette-Impots-Python'}
        response = requests.post(cgi_url, headers=headers, data=saisie_variables)
        root_node = etree.fromstring(response.text, etree.HTMLParser())
        return self.iter_results(root_node)

    def iter_results(self, root_node):
        ignored_input_hidden_names = (
            'blanc',  # white line only used for presentation
            'mesgouv2',  # explanations text block
            )
        result = {}
        for element in root_node.xpath('//input[@type="hidden"][@name]'):
            element_name = element.get('name').strip()
            if element_name in ignored_input_hidden_names:
                continue
            parent = element.getparent()
            parent_tag = parent.tag.lower()
            if parent_tag == 'table':
                tr = parent[parent.index(element) - 1]
                # assert tr.tag.lower() == 'tr', tr
            else:
                # assert parent_tag == 'tr', parent_tag
                tr = parent
            while True:
                description = etree.tostring(tr[1]).strip().strip(u'*').strip()
                if description:
                    break
                table = tr.getparent()
                tr = table[table.index(tr) - 1]
            result[element_name] = float(element.get('value').strip().strip(u'*'))
        print 'Online IINET= ' + repr(int(result['IINET']))
        return result

class CalculatorComparator():
    def generate_random_sample(self, simple_input_variables_with_range):
        random_sample = {}
        for var in simple_input_variables_with_range:
            random_sample[var] = random.randrange(simple_input_variables_with_range[var][0],
                                                  simple_input_variables_with_range[var][1] + 1)
        return random_sample

    def approx_equal(self,a, b):
        return b < (1 + ERROR_MARGIN) * a and b > (1 - ERROR_MARGIN) * a

    def percent_match(self, v1, v2):
        if v1 == 0 and v2 == 0:
            return 1
        if (v1 == 0 and v2 != 0) or (v2 == 0 and v1 != 0):
            return 0
        return min(float(v1) / float(v2), float(v2) / float(v1))

    def create_empty_results(self, results1, results2):
        empty_results = {}
        for var1 in results1[0]:
            empty_results[var1] = {}
            for var2 in results2[0]:
                empty_results[var1][var2] = 0
        return empty_results

    def compute_correlations(self, results1, results2):
        assert len(results1) == len(results2)
        correlations = self.create_empty_results(results1, results2)
        total_amounts = self.create_empty_results(results1, results2)
        for i in range(0, len(results1)):
            for var1 in results1[i]:
                if var1 in correlations:
                    for var2 in results2[i]:
                        if var2 in correlations[var1]:
                            correlations[var1][var2] += self.percent_match(results1[i][var1], results2[i][var2])
                            total_amounts[var1][var2] += (results1[i][var1] + results2[i][var2]) / 2
                    # if approx_equal(results1[i][var1], results2[i][var2]):
                    # results[var1][var2] += 1
        return correlations, total_amounts


    def print_results(self, correlations, total_amounts, nb_tests, base_results, compared_results, only_impot):
        def displaying(var):
            if only_impot:
                return var in ['irpp', 'IINET (Total de votre imposition )',
                               'IRN (Impot net ou restitution nette )']
            else:
                return True

        base_average = self.compute_average_results(base_results)
        compared_average = self.compute_average_results(compared_results)

        # Normalize the results compared to the number of tests
        for var1 in correlations:
            association = {}
            for var2 in correlations[var1]:
                if total_amounts[var1][var2] > 0:
                    association[var2] = correlations[var1][var2] / nb_tests
            sorted_associations = sorted(association.items(), key=operator.itemgetter(1), reverse=True)

            if displaying(var1):
                print ('{} = ~{}'.format(var1, base_average.get(var1, 0)))

                if len(sorted_associations) == 0:
                    print ('        All variables are null, no correlations found ')
                    continue
            else:
                continue

            """
                If we find the same name of variable, we compare them directly.
                Otherwise we find the closest
            """
            displayed = []
            top_value = sorted_associations[0][1]
            if var1 in association and displaying(var1):
                displayed.append(var1)
                print ('        ' + var1 + ' ( ~ '+ str(compared_average.get(var1, 0)) + ' & ' + str(100 * association[var1]) + ' % match )')
            for elem in sorted_associations[0:PRINTING_MAX_LINES]:
                if elem[1] > PRINTING_THRESHOLD * top_value and displaying(elem[0]) and elem[0] not in displayed:
                    displayed.append(elem[0])
                    print ('        ' + elem[0] + ' ( ~ ' + str(compared_average.get(elem[0], 0)) + ' & ' + str(100 * elem[1]) + ' % match )')

    """
        Openfisca should always be simulated before M and Online because Openfisca is based on brut salary
    """
    def simulate_of(self, test_cases):
        print ('Simulating OF')
        timeBeforeOF = time.time()
        of_simulator = OpenfiscaSimulator(test_cases)
        result = of_simulator.simulate(verbose=False)
        diff = time.time() - timeBeforeOF
        print 'Openfisca took {} seconds'.format(diff)
        return result

    def simulate_m(self, test_cases, m_descriptions):
        print ('Simulating M')
        timeBeforeM = time.time()
        m_simulator = MSimulator(test_cases, m_descriptions)
        result = m_simulator.simulate(verbose=False)
        diff = time.time() - timeBeforeM
        print 'M took {} seconds'.format(diff)
        return result

    def simulate_online(self, test_cases, m_descriptions):
        print ('Simulating Online')
        timeBeforeOL = time.time()
        ol_simulator = OnlineTaxSimulator(test_cases, m_descriptions)
        result = ol_simulator.simulate(verbose=False)
        diff = time.time() - timeBeforeOL
        print 'Online took {} seconds'.format(diff)
        return result

    def compute_average_results(self, results):
        summed_results = {}
        for result in results:
            for key in result:
                summed_results[key] = summed_results.get(key, 0) + result[key]
        for key in summed_results:
            summed_results[key] = summed_results[key] / len(results)
        return summed_results

    def save_as_json(self, name, objectname):
        with open('../results/' + name, "w") as outfile:
            json.dump(objectname, outfile)

    def load_from_json(self, filename):
        with open('../results/' + filename, 'r') as f:
            return json.load(f)

    def load_openfisca_from_json(self, filename):
        self.results_openfisca = self.load_from_json(filename + '-openfisca.json')
        self.testcases = self.load_from_json(filename + '-testcases.json')

    def load_results_from_json(self, filename):
        self.results_openfisca = self.load_from_json(filename + '-openfisca.json')
        self.results_m = self.load_from_json(filename + '-m.json')
        self.results_online = self.load_from_json(filename + '-online.json')
        self.testcases = self.load_from_json(filename + '-testcases.json')

    def compute_correlations_openfisca_m_online(self, test_cases, only_impot=True, save=None, load=None):
        m_descriptions = MDescriptions()

        if load:
            results_openfisca = self.load_from_json(load + '-openfisca.json')
            results_m = self.load_from_json(load + '-m.json')
            results_online = self.load_from_json(load + '-online.json')
            test_cases = (load + '-testcases.json', test_cases)
        else:
            results_online = self.simulate_online(test_cases, m_descriptions)
            results_openfisca = self.simulate_of(test_cases)
            results_m = self.simulate_m(test_cases, m_descriptions)

        if save:
            self.save_as_json(save + '-openfisca.json', results_openfisca)
            self.save_as_json(save + '-m.json', results_m)
            self.save_as_json(save + '-online.json', results_online)
            self.save_as_json(save + '-testcases.json', test_cases)

        print ('\n\n ONLINE - M \n')
        correlations, total_amounts = self.compute_correlations(results_online, results_m)
        self.print_results(correlations, total_amounts, len(results_openfisca), results_online, results_m, only_impot)

        print ('\n\n OF - M \n')
        correlations, total_amounts = self.compute_correlations(results_openfisca, results_m)
        self.print_results(correlations, total_amounts, len(results_openfisca), results_openfisca, results_m, only_impot)

        print ('\n\n OF - ONLINE \n')
        correlations, total_amounts = self.compute_correlations(results_openfisca, results_online)
        self.print_results(correlations, total_amounts, len(results_openfisca), results_openfisca, results_online, only_impot)

        return results_openfisca, results_m, results_online

    def openfisca_vs_impotsgouv(self, test_cases):
        m_descriptions = MDescriptions()
        openfisca = self.simulate_of(test_cases)
        online = self.simulate_online(test_cases, m_descriptions)
        return openfisca, online

    def openfisca_vs_m(self, test_cases):
        m_descriptions = MDescriptions()
        openfisca = self.simulate_of(test_cases)
        m = self.simulate_m(test_cases, m_descriptions)
        return openfisca, m

    def m_vs_impotsgouv(self, test_cases):
        m_descriptions = MDescriptions()
        online = self.simulate_online(test_cases, m_descriptions)
        m = self.simulate_m(test_cases, m_descriptions)
        return m, online

    def get_values_from(self, results, variable_name):
        return list(result.get(variable_name, 0) for result in results)

    def get_variable_from_m(self, variable_name):
        return self.get_variable_from_list(variable_name, self.results_m)

    def get_variable_from_online(self, variable_name):
        return self.get_variable_from_list(variable_name, self.results_online)

    def get_variable_from_openfisca(self, variable_name):
        return self.get_variable_from_list(variable_name, self.results_openfisca)

    def get_variable_from_testcases(self, variable_name):
        return self.get_variable_from_list(variable_name, self.testcases)

    def get_variable_from_list(self, variable_name, list_of_results):
        try:
            return self.get_values_from(list_of_results, variable_name)
        except KeyError as e:
            print 'Warning: key not found. Available keys are: ' + repr(list_of_results[0].keys())
        except AttributeError:
            print 'No results loaded. Get some with "compute_correlations_openfisca_m_online" or load some with "load_results_from_json"'

    def get_variable_from_testcases(self, variable_name):
        try:
            return self.get_values_from(self.testcases, variable_name)
        except KeyError as e:
            print 'Key not found. Available keys are: ' + repr(self.testcases[0].keys())


    def compare_results(self, test_cases, x_axis, results1, var1, results2, var2):
        r1 = []
        r2 = []
        diff = []
        x = []
        for i in range(0, len(results1)):
            r1.append(results1[i][var1])
            r2.append(results2[i][var2])
            diff.append(r2[-1] - r1[-1])
            x.append(test_cases[i][x_axis])
        combined = r1 + r2
        colors = ['blue'] * len(r1) + ['green'] * len(r2)
        return x, r1, r2, diff, combined, colors

    def plot_comparison(self, x_data, x_variable, y1_data, y1_variable, y2_data, y2_variable):
        import matplotlib as plt
        x, r1, r2, diff, combined, colors = self.compare_results(
            x_data,x_variable,
            y1_data, y1_variable,
            y2_data, y2_variable)
        plt.scatter(x, r1, c=colors)

def main():
    global parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', default='10', type=int, help='The number of tests')
    parser.add_argument('--ir', default=False, type=bool, help='If we only compute the impot sur le revenu')
    parser.add_argument('--save', default=None, type=str, help='Saves the result in a json file')
    parser.add_argument('--load', default=None, type=str, help='Loads from a json file')
    parser.add_argument('--linear', default=False, type=bool, help='If we generate from a linear distribution instead of gaussian')

    args = parser.parse_args()

    comparator = CalculatorComparator()
    test_cases = CerfaPopulationSimulator().generate_test_cases(args.tests, linear=args.linear)
    comparator.compute_correlations_openfisca_m_online(test_cases, only_impot=args.ir, save=args.save, load=args.load)

    return 0


if __name__ == "__main__":
    sys.exit(main())
