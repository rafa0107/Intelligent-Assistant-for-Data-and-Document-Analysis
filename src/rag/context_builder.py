def context_builder(max_chars, metadata_list, limit) ->str:
    max_chars = 800

    lista_final = sorted(
        metadata_list,
        key=lambda x: x["score"],
        reverse=True
    )

    contexto = ""

    if len(lista_final) <= limit:
        for i in range(len(lista_final)):
            text_limited = lista_final[i]["text"][:max_chars]

            contexto += (
                f"[Contexto {i+1} | "
                f"Similaridade: {lista_final[i]['score']:.03f}]\n"
                f"{text_limited}\n\n"
            )

    else:
        for i in range(limit):
            text_limited = lista_final[i]["text"][:max_chars]

            contexto += (
                f"[Contexto {i+1} | "
                f"Similaridade: {lista_final[i]['score']:.03f}]\n"
                f"{text_limited}\n\n"
            )

    return contexto

def top_k_index(top_k : int, scores : list)->list:
    top_indices = scores[-top_k:][::-1] 
    return top_indices