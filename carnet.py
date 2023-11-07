from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Alexander Bareli

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves")
    ]
)

# Defining the parameters using CPT

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ['Works', "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ['Full', "Empty"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no']}
)

# Associating the parameters with the model structure
car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)
car_infer = VariableElimination(car_model)


def main():
    # Given that the car will not move, what is the probability that the battery is not working? (35%)
    print("\n Given that the car will not move, what is the probability that the battery is not working?")
    no_move_battery = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(no_move_battery, '\n')

    # Given that the radio is not working, what is the probability that the car will not start? (86%)
    print("\n Given that the radio is not working, what is the probability that the car will not start?")
    no_radio_start = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(no_radio_start, '\n')

    # Given that the battery is working, does the probability of the radio working change if we discover that the car has
    # gas in it? (No, the probability does not change)
    print("\n Given that the battery is working, does the probability of the radio working change if we discover that the "
          "car has gas in it?")
    battery_radio = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    battery_radio_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print(battery_radio)
    print(battery_radio_gas)

    # Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car
    # does not have gas in it? (The probability of the ignition failing if we don't have gas is less likely than if we
    # don't know how much gas we have (48% - no gas vs 56% - unknown gas))
    print("\n Given that the car doesn't move, how does the probability of the ignition failing change if we observe that "
          "the car does not have gas in it?")
    no_move_ignition = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    no_move_ignition_gas = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print(no_move_ignition)
    print(no_move_ignition_gas)

    # What is the probability that the car starts if the radio works, and it has gas in it? (72%)
    print("\n What is the probability that the car starts if the radio works and it has gas in it?")
    radio_gas_start = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print(radio_gas_start)

    cpd_key = TabularCPD(
        variable="KeyPresent",
        variable_card=2,
        values=[[0.7], [0.3]],
        state_names={"KeyPresent": ["yes", "no"]}
    )

    cpd_starts2 = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"],
                     "KeyPresent": ['yes', 'no']}
    )

    car_model2 = BayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition", "Starts"),
            ("Gas", "Starts"),
            ("KeyPresent", "Starts"),
            ("Starts", "Moves")
        ]
    )

    car_model2.add_cpds(cpd_starts2, cpd_key, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)
    car2_infer = VariableElimination(car_model2)
    print("\n Probability of car starting if key is present, ignition works and gas is full")
    print(car2_infer.query(variables=["Starts"], evidence={"KeyPresent": "yes", "Ignition": "Works", "Gas": "Full"}))
    print("\n Probability of car starting if key is missing, ignition doesn't work and gas is empty")
    print(car2_infer.query(variables=["Starts"], evidence={"KeyPresent": "no", "Ignition": "Doesn't work",
                                                           "Gas": "Empty"}))
