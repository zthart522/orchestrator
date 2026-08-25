import json
from orchestrator.target_property.factory import target_property_builder
from orchestrator.storage.factory import storage_builder
from orchestrator.scheduler.factory import scheduler_builder

with open('test_inputs/copper_standardpress_melting_input.json') as f:
    config = json.load(f)

tp_config = config['target_property']

melting_point = target_property_builder.build(
    tp_config['target_property_type'],  # 'MeltingPoint'
    tp_config['target_property_args'],  # the nested dict
)

scheduler = scheduler_builder.build(
    config['scheduler']['scheduler_type'],
    config['scheduler']['scheduler_args'],
)

storage = storage_builder.build(
    config['storage']['storage_type'],
    config['storage']['storage_args'],
)

print(vars(melting_point))
print(scheduler.root_directory)

# run the calculation
# result = melting_point.calculate_property(
#    scheduler=scheduler,
#    storage=storage,
#    **tp_config['calculate_property_args'],
# )

# print(result)
