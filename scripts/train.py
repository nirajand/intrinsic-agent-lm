import argparse
from ialm.pipeline import TrainingPipeline
p=argparse.ArgumentParser(); p.add_argument("recipe"); args=p.parse_args(); TrainingPipeline(args.recipe).run()
