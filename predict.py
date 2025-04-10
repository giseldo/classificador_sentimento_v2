import joblib

def predict_sentiment(text):
    model = joblib.load("model.pkl")
    prediction = model.predict([text])
    return prediction[0]

if __name__ == '__main__':
    texts = [
        "Adorei o filme, muito bom!",
        "Filme horrível, perdi meu tempo.",
        "Mais ou menos, esperava mais.",
        "Excelente, recomendo a todos!",
        "Não gostei do final.",
        "A atuação dos atores foi incrível.",
        "Roteiro fraco e sem graça.",
        "Me surpreendeu positivamente.",
        "Um dos piores filmes que já vi.",
        "Comédia muito divertida.",
        "A história é envolvente do início ao fim.",
        "Efeitos especiais impressionantes.",
        "Diálogos mal escritos e artificiais.",
        "Trilha sonora emocionante.",
        "Fotografia impecável.",
        "Direção confusa e amadora.",
        "Atuações medianas.",
        "Um filme para refletir.",
        "Não consegui entender a mensagem.",
        "Previsível e clichê.",
        "Uma obra-prima!",
        "Me fez chorar do começo ao fim.",
        "Suspense de tirar o fôlego.",
        "Romance açucarado e enjoativo.",
        "Ação desenfreada e sem sentido.",
        "Um filme para toda a família.",
        "Não recomendo para menores de 18.",
        "Uma experiência cinematográfica única.",
        "Me senti entediado durante todo o filme.",
        "Um clássico instantâneo.",
        "Amei cada segundo!",
        "Que decepção!",
        "Vale cada centavo do ingresso.",
        "Não perca seu tempo com isso.",
        "Um filme que te faz pensar.",
        "Atuação impecável da protagonista.",
        "Final surpreendente e inesperado.",
        "Um filme que marcou minha vida.",
        "Com certeza verei de novo.",
        "Não entendi o hype em cima desse filme.",
        "Um dos melhores do ano!",
        "Me arrependi de ter assistido.",
        "Uma história inspiradora.",
        "Paisagens deslumbrantes.",
        "Um filme para rir e chorar.",
        "Não me canso de assistir.",
        "Um filme que te prende do início ao fim.",
        "Elenco talentosíssimo.",
        "Um filme que te faz sonhar.",
        "Simplesmente perfeito!"
    ]

    for text in texts:
        sentiment = predict_sentiment(text)
        print(f"Texto: {text}\nSentimento: {sentiment}\n")
