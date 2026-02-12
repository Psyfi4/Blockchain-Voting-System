from aadhaar_system import AadhaarFaceSystem

def main():
    system = AadhaarFaceSystem()

    while True:
        print("\n1. Register Person (Live)")
        print("2. Recognize Person (Live)")
        print("3. Exit")

        ch = input("Choice: ")

        if ch == "1":
            system.register_live()
        elif ch == "2":
            system.recognize_live()
        elif ch == "3":
            break

if __name__ == "__main__":
    main()
