from config import DEVICE, ARCH_NAME, WEIGHT_PATHS
from inference import load_binary_models, example_single_inference


def main() -> None:
    models_dict = load_binary_models(
        arch_name=ARCH_NAME,
        weight_paths=WEIGHT_PATHS,
        device=DEVICE,
    )
    example_single_inference(models_dict)


if __name__ == "__main__":
    main()
