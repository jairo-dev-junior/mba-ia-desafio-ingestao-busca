from search import answer_question


def main():
    print("Faça sua pergunta:")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            return

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            return

        if not question:
            print("RESPOSTA: Digite uma pergunta valida.")
            continue

        try:
            answer = answer_question(question)
        except Exception as exc:
            print(f"RESPOSTA: Erro ao processar a pergunta: {exc}")
            continue

        print(f"RESPOSTA: {answer}")

if __name__ == "__main__":
    main()
