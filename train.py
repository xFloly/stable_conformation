import warnings

from utils.config import get_config
from utils.training_loop import training_loop

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    training_loop(config)