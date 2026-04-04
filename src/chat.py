from search import search_prompt


def main():
    try:
        chain = search_prompt()
        # Fail fast before opening the interactive loop when the backing collection is missing.
        if chain.is_empty():
            print("A coleção está vazia. Execute a ingestão do PDF antes de iniciar o chat.")
            return
    except KeyboardInterrupt:
        print()
        return
    except Exception as e:
        print(f"Não foi possível iniciar o chat: {e}")
        return

    print("Faça sua pergunta:")

    while True:
        try:
            question = input("PERGUNTA: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if question.lower() in {"sair", "exit", "quit"}:
            return

        if not question.strip():
            continue

        if len(question) > 500:
            print("Pergunta acima de 500 caracteres; truncando para 500.")
            question = question[:500]

        try:
            response = chain(question)
        except KeyboardInterrupt:
            print()
            return
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
            continue

        print(f"RESPOSTA: {response}")

if __name__ == "__main__":
    main()
