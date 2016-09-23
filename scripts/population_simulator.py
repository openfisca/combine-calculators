# -*- coding: UTF-8 -*-

"""
    Simulate a population of N inhabitants from the statistics on the tax declaration.
"""

"""
ONLINE CALCULATOR ASKS FOR:

0AX:
0AY:
0AZ:
0AH:
simplifie:1
pre_0AXJ:
pre_0AXM:
pre_0AAJ:
pre_0AAM:
pre_situation_famille:C
pre_0AYJ:
pre_0AYM:
pre_0AZJ:
pre_0AZM:
pre_situation_residence:M
0DA:1983
0DB:
0CF:
0F0:
0F1:
0F2:
0F3:
0F4:
0F5:
0CG:
0G0:
0G1:
0G2:
0CH:
0H1:
0H2:
0H3:
0H4:
0H5:
0H6:
0CI:
0I1:
0I2:
0I3:
0CR:
0R0:
0R1:
0R2:
0DJ:
0DN:
1AJ:19000
1BJ:
1CJ:
1DJ:
1AP:
1BP:
1CP:
1DP:
1AK:
1BK:
1CK:
1DK:
1AV:
1BV:
1CV:
1DV:
1BL:
1CB:
1DQ:
1AS:
1BS:
1CS:
1DS:
1AT:
1BT:
1AZ:
1BZ:
1CZ:
1DZ:
1AO:
1BO:
1CO:
1DO:
1AW:
1BW:
1CW:
1DW:
2DH:
2EE:
2DC:
2FU:
2CH:
2TS:
2GO:
2TR:
2FA:
2CG:
2BH:
2CA:
2AB:
2CK:
2BG:
2LA:
2LB:
2AA:
2AL:
2AM:
2AN:
2AQ:
2AR:
2DM:
3VG:
3VH:
3SG:
3SH:
4BE:
4BA:
4BB:
4BC:
4BD:
4BF:
0XX:
6DE:
6GI:
6GJ:
6EL:
6EM:
6GP:
6GU:
6DD:
6RS:
6SS:
6RT:
6ST:
6RU:
6SU:
APS:
BPS:
APT:
BPT:
APU:
BPU:
CPS:
DPS:
CPT:
DPT:
CPU:
DPU:
6QS:
6QT:
6QU:
7UD:
7UF:
7UH:
7XS:
7XT:
7XU:
7XW:
7XY:
7VA:
7VC:
7AC:
7AE:
7AG:
7DB:
7DF:
7DD:
7DL:
7VZ:
7VV:
7VU:
7VT:
7VX:
7CD:
7CE:
7GA:
7GE:
7GB:
7GF:
7GC:
7GG:
7EA:
7EB:
7EC:
7ED:
7EF:
7EG:
7GZ:
7UK:
7VO:
7TD:
7WN:
7WO:
7WM:
7WP:
8YS:
8YU:
7SD:
7SA:
7SE:
7SB:
7SF:
7SC:
7WC:
7WB:
7SG:
7RG:
7VG:
7VH:
7SH:
7RH:
7SI:
7RI:
7WT:
7WU:
7SJ:
7RJ:
7SK:
7RK:
7SL:
7RL:
7SN:
7RN:
7SP:
7RP:
7SR:
7RR:
7SS:
7RS:
7SQ:
7RQ:
7ST:
7RT:
7SV:
7TV:
7SW:
7TW:
7RV:
7RW:
7RZ:
7WJ:
7WL:
8BY:
8CY:
8UT:
8TF:
8TI:
8TK:
"""

from operator import itemgetter
import random
import argparse
import sys

class CerfaPopulationSimulator():
    def __init__(self, model="2014"):
        if (model == "2014"):
            self.possible_situations = [('M', 12002841), ('D', 5510809), ('O', 983754), ('C', 14934477), ('V', 3997578)]
            self.children_per_family = [(1, 46), (2, 38.5), (3, 12.5), (4, 2), (5, 1)]
            self.families = 9321480
            self.salaire_imposable = [21820704, 503723963299]

        self.total_declarations = float(sum(w for c, w in self.possible_situations))
        self.avg_salaire_imposable = self.salaire_imposable[1] / self.salaire_imposable[0]
        self.percent_salaire_imposable_not_0 = self.salaire_imposable[0] / self.total_declarations

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

    def generate_random_cerfa(self, linear=False):
        cerfa = {}
        # Birthdate is simulated from age that is random between 18 and 88
        cerfa['0DA'] = 2014 - int(random.random() * 70 + 18)
        # # Drawing the situation
        situation = self.weighted_choice(self.possible_situations)
        cerfa[situation] = 1

        ## We only give children to married or pacces. This is an approximation
        enfants = 0
        if situation == 'M' or situation == 'O':
            # If marie or pacse we have someone else in the familly who has a salary sometimes
            cerfa['1BJ'] = int(self.simulate_one_gaussian(mu=18675, sigma=11156, th=0.187))
            cerfa['0DB'] = 2014 - int(random.random() * 70 + 18)

            if random.random() < (self.families / float(self.possible_situations[0][1] + self.possible_situations[2][1]) - 0.05):
                enfants = self.weighted_choice(self.children_per_family)
        else:
            if random.random() < 0.05:
                enfants = self.weighted_choice(self.children_per_family)

        if enfants > 0:
            cerfa['F'] = enfants

        if linear:
            cerfa['1AJ'] = random.random() * 100000
        else:
            cerfa['1AJ'] = int(self.simulate_one_gaussian(mu=21589, sigma=12606, th=0.612))

        return cerfa

    # This is a gradient desent to find the optimal parameters
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

    def generate_test_cases(self, nb_test, linear=False):
        test_cases = []
        for test in range(0, nb_test):
            test_case = self.generate_random_cerfa(linear=linear)
            test_case['year'] = 2014
            test_cases.append(test_case)
        sorted_tests = sorted(test_cases, key=itemgetter('1AJ'))

        print 'Generating test cases:'
        for case in sorted_tests:
            print repr(case)
        return sorted_tests

def main():
    global parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filled', default='0', type=int, help='The number of people matching')
    parser.add_argument('--total', default='0', type=int, help='The total value for those people')

    args = parser.parse_args()

    simulator = CerfaPopulationSimulator()
    simulator.find_gaussian_parameters(args.filled, args.total)
    return 0

if __name__ == "__main__":
    sys.exit(main())
