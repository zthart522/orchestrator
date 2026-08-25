import json
from orchestrator.target_property.factory import target_property_builder
from orchestrator.scheduler.factory import scheduler_builder

with open('./test_inputs/'
          'water_ethanol_solvation_free_energy_input.json', 'r') as f:
    config = json.load(f)

tp_config = config['target_property']

solvation_free_energy = target_property_builder.build(
    tp_config['target_property_type'],  # 'MeltingPoint'
    tp_config['target_property_args'],  # the nested dict
)

scheduler = scheduler_builder.build(
    config['scheduler']['scheduler_type'],
    config['scheduler']['scheduler_args'],
)

x = solvation_free_energy.calculate_property(
    **tp_config['calculate_property_args'], scheduler=scheduler)

print(f"SOLVATION FREE ENERGY RESULT: {x['property_value']}")
