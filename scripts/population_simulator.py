# -*- coding: UTF-8 -*-

"""
    Simulate a population of N inhabitants from the statistics on the tax declaration.
"""

import random

class CerfaPopulationSimulator():
    def __init__(self, model="2014"):
        if (model == "2014"):
            self.possible_situations = [('M', 12002841), ('D', 5510809), ('O', 983754), ('C', 14934477), ('V', 3997578)]
            self.children_per_family = [(1, 46), (2, 38.5), (3, 12.5), (4, 2), (5, 1)]
            self.families = 9321480
            self.salaire_imposable[21820704, 503723963299]

        self.self.total_declarations = float(sum(w for c, w in self.possible_situations))
        self.avg_salaire_imposable = self.salaire_imposable[1] / self.salaire_imposable[0]
        self.percent_salaire_imposable_not_0 = self.salaire_imposable[0] / self.self.total_declarations

    def weighted_choice(self, choices):
       total = sum(w for c, w in choices)
       r = random.uniform(0, total)
       upto = 0
       for c, w in choices:
          if upto + w >= r:
             return c
          upto += w
       assert False, "Shouldn't get here"

    def simulate_one_gaussian(self, th, mu, sigma):
        if random.random() < th:
            return max(random.gauss(mu, sigma), 0)
        return 0

    def generate_random_cerfa(self):
        cerfa = {}

        # Birthdate is simulated from age that is random between 18 and 88
        cerfa['0DA'] = 2014 - int(random.random() * 70 + 18)

        # # Drawing the situation
        # situation = self.weighted_choice(self.possible_situations)
        # cerfa[situation] = 1
        #
        # ## We only give children to married or pacces. This is an approximation
        # enfants = 0
        # if situation == 'M' or situation == 'O':
        #     if random.random() < (self.families / float(self.possible_situations[0][1] + self.possible_situations[2][1])):
        #         enfants = self.weighted_choice(self.children_per_family)
        # if enfants > 0:
        #     cerfa['F'] = enfants

        # The parameters of the gaussian are obtained by running the function find_gaussian_parameters once
        cerfa['1AJ'] = self.simulate_one_gaussian(mu=21589, sigma=12606, th=0.612)

        return cerfa

    def find_gaussian_parameters(self, number_not_0, total_value, distribution_percentage_null=5):
        def gradiant(a, b):
            if a > b:
                return min((a / b - 1) * random.random(), 0.5)
            else:
                return min((b / a - 1) * random.random(), 0.5)

        def simulate_population(th, mu, sigma, percentage_repr):
            total_result = 0
            number_not_null = 0
            for i in range(0, int(self.total_declarations * percentage_repr)):
                result = self.simulate_one_gaussian(th, mu, sigma)
                total_result += result
                if result > 0:
                    number_not_null += 1
            return number_not_null / percentage_repr, total_result / percentage_repr

        # Between 0 and 1
        number_not_0 = float(number_not_0)
        total_value = float(total_value)

        percentage_repr = 0.001
        mu = total_value / number_not_0
        sigma = mu / 2
        mu_step = mu / 2
        sigma_step = sigma / 2
        th = (1 + distribution_percentage_null / 100.0) * number_not_0 / self.total_declarations
        print repr(th)
        max_number_of_simulations = 100
        for i in range(0, max_number_of_simulations):
            sim_not_0, sim_tot_value = simulate_population(th, mu, sigma, percentage_repr)
            if sim_not_0 > number_not_0:
                mu -= mu_step * gradiant(number_not_0, sim_not_0)
                # sigma -= sigma_step * gradiant(number_not_0, sim_not_0)
            else:
                mu += mu_step * gradiant(sim_not_0, number_not_0)
                # sigma += sigma_step * gradiant(sim_not_0, number_not_0)

            if sim_tot_value > total_value:
                # mu -= mu_step * gradiant(total_value, sim_tot_value)
                sigma -= sigma_step * gradiant(total_value, sim_tot_value)
            else:
                # mu += mu_step * gradiant(sim_tot_value, total_value )
                sigma += sigma_step * gradiant(sim_tot_value, total_value)
            print 'Total target ' + str(sim_tot_value/total_value) + ' not 0 target: ' + str(sim_not_0/number_not_0) + ' mu=' +  repr(mu) + ', sigma=' + repr(sigma) + ', th=' + str(th)
            mu_step = mu_step * 0.995
            sigma_step = sigma_step * 0.995
            percentage_repr = percentage_repr * 1.01

    def generate_test_cases(self, nb_test):
        test_cases = []
        for test in range(0, nb_test):
            test_case = self.generate_random_cerfa()
            test_case['year'] = 2014
            test_cases.append(test_case)
            print repr(test_case)
        return test_cases
