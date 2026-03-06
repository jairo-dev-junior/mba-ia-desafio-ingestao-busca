import argparse

from langchain_core.messages import HumanMessage

from rag import OUT_OF_CONTEXT_ANSWER, get_llm, get_vector_store


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _format_context(results):
    if not results:
        return ""

    formatted_chunks = []
    for index, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "document.pdf")
        page = doc.metadata.get("page")
        location = f"{source} - pagina {page + 1}" if page is not None else source
        formatted_chunks.append(
            f"[Trecho {index} | score={score:.4f} | {location}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def answer_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("A pergunta nao pode estar vazia.")

    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=10)
    context = _format_context(results)

    if not context:
        return OUT_OF_CONTEXT_ANSWER

    prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=question)
    response = get_llm().invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", "") or ""

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        )

    answer = str(content).strip()
    return answer or OUT_OF_CONTEXT_ANSWER


def search_prompt(question=None):
    if question is None:
        return answer_question
    return answer_question(question)


def main():
    parser = argparse.ArgumentParser(description="Busca semantica no PDF indexado.")
    parser.add_argument("question", nargs="*", help="Pergunta a ser respondida.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("PERGUNTA: ").strip()

    if not question:
        raise ValueError("A pergunta nao pode estar vazia.")

    print(answer_question(question))


if __name__ == "__main__":
    main()
