# combine-calculators

Experimental! Use OpenFisca with other calculators, compare results, merge their formulas
Scripts files and notebooks are in scripts/

# Running calculators against each other

TL;DR Compares M, Openfisca, and Online on a given population by running:

python simulate_comparators --tests=2000 --save='my_comparison'

1) in population_simulator.py a population of N elements is created in CERFA formalism
2) in input_variable_converter.py methods are used convert this population in M, Openfisca, and Online formalism
3) simulations are run on all three comparators
4) a parameter automatic matching is displayed in command line
5) results are saved to json if a --save argument is given

Optional: 
  --ir: only compute "impot sur le revenu":  
  --load=file: loads from a file

# Visual comparison of results 

visual_comparisons.ipynb propose a vizualization of the comparisons with graphs and filterable worse cases.

# Tax Legislation Refactoring

The tax/benefit legislation on individuals in France is more than 200,000 lines long.

We believe that with this tool, you can make it less than 100 lines, transparent, and 95% similar to the existing legislation

### How it works:

1. Define your concepts (e.g., 'nb_enfants', 'age', 'parent isolé') and budget (e.g.: cost less than 2 millions euros)
2. The machine learning algorithm helps you adjust the parameters of your reform to approximate the current legislation and fit your budget
3. From the biggest discrepencies with the current legislation, you can improve your concepts (or the legislation)
4. Repeat until you reach a legislation that matches your own goals. The algorithm takes care of minimizing your budget and maximizing the similarity to current legislation.

### Beta version limitations:

For this test, we only take a population of people from all ages, who have 0 to 5 children, no salary, married, veuf, paces, divorces or celibataires, and simulate the "aides sociales"

### Current Results:

Within a few minutes, we got a tax reform for "aides sociales" for people with no salary that is:

* 6 lines long 
* similar to the existing in average at more than 95%


# Adding more "cases CERFA" to make the comparator more precise!

For now, we only cover about 10 most commun CERFA inputs, including situation, children, salary.
Adding more CERFA inputs will make the comparator more powerful and reliable. 

1. start adding the CERFA input with its distribution in population_simulator.py 

If you do not know the exact distribution, we wrote a function to get an approximation from the values provided in: http://www2.impots.gouv.fr/documentation/statistiques/2042_nat/2015/revenus_2014_6e_ano.pdf

a) get the value_filled and total_amount for your input
b) run:
python population_simulator.py --filled=value_filled --total=total_amount
c) it will give you the parameters that you can put in in the function "generate_random_cerfa" of "population_simulator.py"

2. CERFA inputs get converted automatically to OpenFisca, M, and Online

If the "alias" of the CERFA input is filled in OpenFisca, all conversion to M, OpenFisca and Online should be automatic.
When the alias is not filled in, you might need to update input_variable_converter.py.

3. Run calculators against each other (cf above)

4. Vizualize results (cf above)
  
