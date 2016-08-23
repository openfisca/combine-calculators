# -*- coding: UTF-8 -*-

"""
    Tool for converting a CERFA form into an OpenFisca situation
"""

import openfisca_france
import json
from openfisca_trace_explorer import OpenfiscaTraceExplorer
from m_compute import m_compute_from_aliases


# Example of usage:

# converter = CerfaOpenFiscaConverter()
# openfisca_inputs = converter.convert_cerfa_inputs_to_openfisca(example_cerfa_variables, period=2014)


# Example of cerfa variables:

# example_cerfa_variables = {
#     ## Declarant 1, Declarant 2
#     '0DA': 1981, '0DB': 1948,
#
#     ## Parent1 Parent2 Dependant1 Dependant2
#
#     ## Indiquez vos seuls revenus d'activité (salaires, droits d'auteur, avantages en nature et indemnités journalières)
#     '1AJ': 250000, # '1BJ': 2000, '1CJ': 8000, '1DJ': 2000,
#
#     ## Indiquez vos autres revenus imposables (indemnité de préretraite, allocation chômage...)
#     # '1AP': 2000, '1BP': 1000, '1CP': 26, '1DP': 0,
#
#     ## Indiquez vos frais réels
#     # '1AK':50, '1BK': 1234, '1CK': 90, '1DK': 1233,
#
#     ## Demandeur d'emploi inscrit depuis plus d'un an
#     # '1AI':0, '1BI':1, '1CI': 0, '1DI': 1,
#
#     ## Travail à temps plein toute l'année 2014 cochez la case
#     # '1AX': 1, '1BX': 0,  '1CX': 0, '1DX': 1,
#     # '1AV': 30, '1BV': 50, '1CV': 50, '1DV': 50,
#
#     # '1BL':3, '1CB':5, '1DQ':7,
#     # '1AS', '1BS', '1CS', '1DS', '1AT', '1BT', '1AZ', '1BZ',
#     # '1CZ', '1DZ', '1AO', '1BO', '1CO', '1DO', '1AW', '1BW',
#     # '1CW', '1DW', '2DH', '2EE', '2DC', '2FU', '2CH', '2TS',
#     # '2GO', '2TR', '2FA', '2CG', '2BH', '2CA', '2AB', '2CK',
#     # '2BG', '2LA', '2LB', '2AA', '2AL', '2AM', '2AN', '2AQ',
#     # '2AR', '2DM', '3VG', '3VH', '3SG', '3SH', '4BE', '4BA',
#     # '4BB', '4BC', '4BD', '4BF', '0XX', '6DE', '6GI', '6GJ',
#     # '6EL', '6EM', '6GP', '6GU', '6DD', '6RS', '6SS', '6RT',
#     # '6ST', '6RU', '6SU', '6PS', '6PS', '6PT', '6PT', '6PU',
#     # '6PU', '6PS', '6PS', '6PT', '6PT', '6PU', '6PU', '6QR',
#     # '6QW', '7UD', '7UF', '7XS', '7XT', '7XU', '7XW',
#     # '7XY', '7VA', '7VC', '7AC', '7AE', '7AG', '7DB', '7DF',
#     # '7DD', '7DL', '7DQ', '7DG', '7VZ', '7VV', '7VU', '7VT',
#     # '7VX', '7CD', '7CE', '7GA', '7GE', '7GB', '7GF', '7GC',
#     # '7GG', '7EA', '7EB', '7EC', '7ED', '7EF', '7EG', '7GZ',
#     # '7UK', '7VO', '7TD', '7WN', '7WO', '7WM', '7WP', '7WE',
#     # '7WG', '7SD', '7SA', '7SE', '7SB', '7SF', '7SC', '7WC',
#     # '7WB', '7SG', '7RG', '7VG', '7VH', '7SH', '7RH', '7SI',
#     # '7RI', '7WT', '7WU', '7SJ', '7RJ', '7SK', '7RK', '7SL',
#     # '7RL', '7SN', '7RN', '7SP', '7RP', '7SR', '7RR', '7SS',
#     # '7RS', '7SQ', '7RQ', '7ST', '7RT', '7SV', '7TV', '7SW',
#     # '7TW', '7RV', '7RW', '7RZ', '7WJ', '7WL', '8BY', '8CY',
#     # '8UT', '8TF', '8TI', '0AC', '0AM', '0AD', '0AV', '0AO',
#     # '0CF'
# }


class MissingFieldException(Exception):
    pass

class CerfaOpenFiscaConverter():
    """
        This class enables to convert inputs from CERFA to inputs for Openfisca and the other way round
    """

    def __init__(self, verbose=True):
        self._tax_benefit_system = openfisca_france.FranceTaxBenefitSystem()
        self._verbose = verbose

    def assign_if_present(self, key, new_key, inputs, outputs):
        if key in inputs:
            outputs[new_key] = inputs[key]

    def find_cerfa_columns(self):
        columns = self._tax_benefit_system.column_by_name.values()
        for col in columns:
            if isinstance(col.cerfa_field, unicode):
                try:
                    # Horrible hack because of OF bug to convert to string the CERFA field
                    col.cerfa_field = json.loads(str(col.cerfa_field.replace("u'",'"').replace("'", '"').replace('0:','"0":').replace('1:','"1":').replace('2:','"2":').replace('3:','"3":').replace('4:','"4":').replace('5:','"5":').replace('6:','"6":').replace('7:','"7":').replace('8:','"8":')))
                except ValueError:
                    # print ' No JSON ' + col.cerfa_field + ' ' + repr(ValueError)
                    pass

        dict_columns = list(filter(lambda column: isinstance(column.cerfa_field, dict), columns))
        string_columns = list(filter(lambda column: (isinstance(column.cerfa_field, unicode) or isinstance(column.cerfa_field, str)) and column not in dict_columns,
                            columns))
        return dict_columns, string_columns


    def handle_special_cerfa(self, cerfa, inputs, person_index, person_output):
        """
            Handle CERFA cases that need special conversion to become an openfisca variable
        """

        special_cases = ['0DA', '0DB']

        # Date de naissance
        if cerfa == special_cases[0] and person_index == 0:
            person_output['date_naissance'] = str(inputs[cerfa]) + '-1-1'
            if self._verbose:
                print 'INFO: converting birthdate from "' + str(inputs[cerfa]) + '" to "' + person_output['date_naissance'] + '"'
        if cerfa == special_cases[1] and person_index == 1:
            person_output['date_naissance'] = str(inputs[cerfa]) + '-1-1'
            if self._verbose:
                print 'INFO: converting birthdate from "' + str(inputs[cerfa]) + '" to "' + person_output['date_naissance'] + '"'



    def convert_cerfa_inputs_to_openfisca(self, inputs, period):
        """
            :param inputs: CERFA cases as a dictionary
            :param period: year on which the simulation is done
            :return: OpenFisca situation
        """

        openfisca_inputs = {}

        # Extracting columns from OpenFisca
        dict_columns, string_columns = self.find_cerfa_columns()
        # for column in dict_columns:
        #     print column.name + ' = ' + repr(column.cerfa_field)
        # for column in string_columns:
        #     print column.name + ' = ' + repr(column.cerfa_field)

        # Mapping with CERFA to columns for string CERFA
        temp_cerfa_to_column_name = {}
        for column in string_columns:
            if column.cerfa_field in temp_cerfa_to_column_name:
                temp_cerfa_to_column_name[column.cerfa_field] = 'none'
            else:
                temp_cerfa_to_column_name[column.cerfa_field] = column.name

        # Creating a mapping CERFA -> (column_name, person)
        # person is -1 if it does not apply to anybody specific (for string columns)
        cerfa_to_column_name = {}
        for cerfa in temp_cerfa_to_column_name:
            if temp_cerfa_to_column_name[cerfa] != 'none':
                cerfa_to_column_name[cerfa] = [temp_cerfa_to_column_name[cerfa], -1]

        # MAPPING FOR MULTIPLE FIELDS
        # First, for who the mapping is:
        for i in range(0,10):
            temp_cerfa_to_column_name = {}
            for column in dict_columns:
                if i in column.cerfa_field:
                    if column.cerfa_field[i] in temp_cerfa_to_column_name:
                        temp_cerfa_to_column_name[column.cerfa_field[i]] = 'none'
                    else:
                        temp_cerfa_to_column_name[column.cerfa_field[i]] = column.name
            for cerfa in temp_cerfa_to_column_name:
                if temp_cerfa_to_column_name[cerfa] != 'none':
                    cerfa_to_column_name[cerfa] = [temp_cerfa_to_column_name[cerfa], i]

        # Inputs independant of a person
        for cerfa in inputs:
            if cerfa not in cerfa_to_column_name:
                self.handle_special_cerfa(cerfa, inputs, -1, openfisca_inputs)
            elif cerfa_to_column_name[cerfa][1] == -1:
                openfisca_inputs[cerfa_to_column_name[cerfa][0]] = inputs[cerfa]

        # Parent1
        parent1 = self.make_person(inputs, cerfa_to_column_name, 0)
        if len(parent1.keys()) > 0:
            openfisca_inputs['parent1'] = parent1

        # Parent2
        parent2 = self.make_person(inputs, cerfa_to_column_name, 1)
        if len(parent2.keys()) > 0:
            openfisca_inputs['parent2'] = parent2

        # Enfants
        enfants = []
        for i in range(0, 9):
            enfant = self.make_person(inputs, cerfa_to_column_name, i + 2)
            if len(enfant.keys()) > 0:
                enfants.append(enfant)
        if len(enfants) > 0:
            openfisca_inputs['enfants'] = enfants

        openfisca_inputs['period'] = period
        return openfisca_inputs

    def make_person(self, inputs, cerfa_to_column_name, index):
        person = {}
        for cerfa in inputs:
            if cerfa not in cerfa_to_column_name:
                self.handle_special_cerfa(cerfa, inputs, index, person)
            elif cerfa_to_column_name[cerfa][1] == index:
                person[cerfa_to_column_name[cerfa][0]] = inputs[cerfa]
        return person
