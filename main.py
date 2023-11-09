from model.commands import create_model, predict


def main() -> None:
    while True:
        command = input("Input Command")
        if command == "exit":
            break
        if command == "createmodel":
            create_model()
        if command == "predict":
            predict()


if __name__ == "__main__":
    main()
