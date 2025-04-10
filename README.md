## Classificador de Sentimento

Este é um classificador de sentimento simples que usa scikit-learn, TF-IDF e Naive Bayes.

### Como usar

1.  **Instale as dependências:**

    ```bash
    pip install scikit-learn joblib
    ```

2.  **Execute o script:**

    ```bash
    python sentiment_classifier.py
    ```

    Isso irá treinar o modelo, salvar o modelo treinado em um arquivo chamado `sentiment_model.joblib`, carregar o modelo e, em seguida, prever o sentimento de algumas frases de exemplo.

### Personalização

Você pode personalizar os dados de treinamento modificando as listas `texts` e `labels` no script `sentiment_classifier.py`.

Você também pode usar seus próprios dados para treinar o modelo. Certifique-se de que os dados estejam em um formato semelhante ao dos dados de exemplo.

### Salvando e Carregando o Modelo

O modelo treinado é salvo em um arquivo chamado `sentiment_model.joblib`. Você pode carregar o modelo usando o método `load_model`.

### Requisitos

*   Python 3.6+
*   scikit-learn
*   joblib
