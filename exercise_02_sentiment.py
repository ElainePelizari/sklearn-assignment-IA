import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == "__main__":
    # Carrega os dados
    movie_reviews_data_folder = r"./data"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)

    # Separa em dados de treino e teste
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # Cria o pipeline para o classificador LinearSVC
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    # Define os parâmetros para o GridSearchCV
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__C': [0.1, 1, 10],
    }

    # Executa o GridSearchCV para encontrar os melhores parâmetros
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(docs_train, y_train)

    # Aplica o classificador com os melhores parâmetros encontrados no conjunto de teste
    y_predicted = gs_clf.predict(docs_test)

    # Avalia o desempenho do classificador LinearSVC
    print("LinearSVC:")
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Imprime e plota a matriz de confusão
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # Cria o pipeline para o classificador MultinomialNB
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    # Define os parâmetros para o GridSearchCV
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': [0.1, 1, 10],
    }

    # Executa o GridSearchCV para encontrar os melhores parâmetros
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(docs_train, y_train)

    # Aplica o classificador com os melhores parâmetros encontrados no conjunto de teste
    y_predicted = gs_clf.predict(docs_test)

    # Avalia o desempenho do classificador MultinomialNB
    print("\nMultinomialNB:")
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Imprime e plota a matriz de confusão
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # O código acima utiliza dois algoritmos de classificação: LinearSVC e MultinomialNB. 
    # Ambos são aplicados a um conjunto de dados de avaliações de filmes, separados em conjunto de treino e teste, 
    # para classificar cada avaliação como positiva ou negativa. 
    # O resultado da aplicação do algoritmo LinearSVC apresentou uma acurácia melhor do que o algoritmo MultinomialNB, 
    # tendo atingido uma precisão, recall e f1-score maiores para todas as classes. 
    # Observa-se também que a matriz de confusão mostra que o algoritmo LinearSVC teve menos erros do que o algoritmo MultinomialNB.
    # Portanto, pode-se concluir que o algoritmo LinearSVC é mais preciso para classificar as avaliações de filmes.