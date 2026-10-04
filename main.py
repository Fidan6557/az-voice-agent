from src.llm import LLMAgent


def main():
    agent = LLMAgent()
    print("Aysel (Azərnet müştəri xidməti) ilə söhbət. Çıxmaq üçün 'çıx' yaz.\n")

    while True:
        user_text = input("Siz: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"çıx", "cix", "exit", "quit"}:
            print("Sağ olun!")
            break

        answer = agent.reply(user_text)
        print(f"Aysel: {answer}\n")


if __name__ == "__main__":
    main()