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
import openfisca_france, random, operator, time, requests
import argparse
import sys

from m_compute import m_compute_from_aliases
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

class TaxbenefitSimulater():
    def __init__(self, testcases):
        self._testcases = testcases

    def simulate(self, verbose=False):
        results = []
        result_count = 0
        for test in self._testcases:
            results.append(self.simulate_one_test(test))
            result_count += 1
            if verbose:
                print ('{} simulations done'.format(result_count))
        return results

    def simulate_one_test(self, test):
        raise NotImplementedError


class OpenfiscaSimulator(TaxbenefitSimulater):
    def __init__(self, testcases):
        self._tax_benefit_system = openfisca_france.FranceTaxBenefitSystem()
        self._testcases = testcases

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
        return self._tax_benefit_system.new_scenario().init_single_entity(
            period=v.get('year'),
            parent1=dict(
                age=v.get('year', 2014) - v.get('0DA', 1981),
                salaire_imposable=v.get('1AJ', 0),
                frais_reels=v.get('1AK', 0),
                chomeur_longue_duree=v.get('1AI', 0),
                ppe_tp_sa=v.get('1AX', 0),
                ppe_du_sa=v.get('1AV', 0),
                revenu_assimile_pension=v.get('1AP', 0),
            ),
            # parent2 = dict(
            # age = v.get('year', 2014) - v.get('0DB', 1981),
            # salaire_de_base = v.get('1BJ', 0),
            # frais_reels = v.get('1BK', 0),
            # chomeur_longue_duree = v.get('1BI', 0),
            #     ppe_tp_sa = v.get('1BX', 0),
            #     ppe_du_sa = v.get('1BV', 0),
            #     revenu_assimile_pension = v.get('1BP', 0)
            #     ),
        )

    def simulate_one_test(self, test):
        scenario = self.make_scenario(test)
        simulation = scenario.new_simulation(trace=True)

        # SIMULATION FOR THE TRACE
        traceon = 'revdisp'
        nivvie = simulation.calculate(traceon, '2014', print_trace=False, max_depth=10, show_default_values=False)[0]
        # trace_explorer = OpenfiscaTraceExplorer(simulation)
        # trace_explorer.print_trace(traceon, 2)
        # trace_explorer.save_trace_as_json(traceon, 2)

        irpp = -simulation.calculate('irpp', '2014')[0]
        credits_impot = simulation.calculate('credits_impot', '2014')[0]
        salaire_imposable = simulation.calculate('salaire_imposable', '2014')[0]
        taux_moyen_imposition = simulation.calculate('taux_moyen_imposition', '2014')[0]
        tot_impot = simulation.calculate('tot_impot', '2014')[0]

        # Now that we have the 1AJ, we don't need the salaire_brut anymore
        return {'irpp': irpp, 'credits_impot': credits_impot, 'salaire_imposable': salaire_imposable,
                'taux_moyen_imposition':taux_moyen_imposition, 'tot_impot': tot_impot}

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


class MSimulator(TaxbenefitSimulater):
    def __init__(self, testcases, m_descriptions):
        self._testcases = testcases
        self._m_descriptions = m_descriptions

    def simulate_one_test(self, test):
        """
            Calls the M code and translate the name of the variable to a human readable name
        """
        result = m_compute_from_aliases(test)
        return {self._m_descriptions.get_description(variable): result[variable] for variable in result}


class OnlineTaxSimulator(TaxbenefitSimulater):
    def __init__(self, testcases, m_descriptions):
        self._testcases = testcases
        self._m_descriptions = m_descriptions

    def simulate_one_test(self, test):
        saisie_variables = {
            '0DA': test.get('0DA', 1981),
            '1AJ': int(test.get('1AJ', -1000000)),
            'pre_situation_famille': 'C',
            'pre_situation_residence': 'M',
            'simplifie': '1',
            }
        cgi_url = 'http://www3.finances.gouv.fr/cgi-bin/calc-{}.cgi'.format(test.get('year', 2014))
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
                assert tr.tag.lower() == 'tr', tr
            else:
                assert parent_tag == 'tr', parent_tag
                tr = parent
            while True:
                description = etree.tostring(tr[1]).strip().rstrip(u'*').strip()
                if description:
                    break
                table = tr.getparent()
                tr = table[table.index(tr) - 1]
            result[self._m_descriptions.get_description(element_name)] = float(element.get('value').strip())
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
        result = of_simulator.simulate(verbose=True)
        diff = time.time() - timeBeforeOF
        print 'Openfisca took {} seconds'.format(diff)
        return result


    def simulate_m(self, test_cases, m_descriptions):
        print ('Simulating M')
        timeBeforeM = time.time()
        m_simulator = MSimulator(test_cases, m_descriptions)
        result = m_simulator.simulate(verbose=True)
        diff = time.time() - timeBeforeM
        print 'M took {} seconds'.format(diff)
        return result


    def simulate_online(self, test_cases, m_descriptions):
        print ('Simulating Online')
        timeBeforeOL = time.time()
        ol_simulator = OnlineTaxSimulator(test_cases, m_descriptions)
        result = ol_simulator.simulate(verbose=True)
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

    def compute_correlations_of_m_online(self, test_cases, only_impot=True):
        m_descriptions = MDescriptions()
        results_simulator_1 = self.simulate_of(test_cases)
        results_simulator_2 = self.simulate_m(test_cases, m_descriptions)
        results_simulator_3 = self.simulate_online(test_cases, m_descriptions)

        print ('\n\n ONLINE - M \n')
        correlations, total_amounts = self.compute_correlations(results_simulator_3, results_simulator_2)
        self.print_results(correlations, total_amounts, len(results_simulator_1), results_simulator_3, results_simulator_2, only_impot)

        print ('\n\n OF - M \n')
        correlations, total_amounts = self.compute_correlations(results_simulator_1, results_simulator_2)
        self.print_results(correlations, total_amounts, len(results_simulator_1), results_simulator_1, results_simulator_2, only_impot)

        print ('\n\n OF - ONLINE \n')
        correlations, total_amounts = self.compute_correlations(results_simulator_1, results_simulator_3)
        self.print_results(correlations, total_amounts, len(results_simulator_1), results_simulator_1, results_simulator_3, only_impot)

        return results_simulator_1, results_simulator_2, results_simulator_3

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

def main():
    global parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', default='10', type=int, help='The number of tests')
    parser.add_argument('--basic', default=True, type=bool, help='If we only compute the total of tax')

    args = parser.parse_args()

    comparator = CalculatorComparator()
    test_cases = CerfaPopulationSimulator().generate_test_cases(args.tests)
    comparator.compare_all(test_cases, only_impot=args.basic)

    return 0


if __name__ == "__main__":
    sys.exit(main())
