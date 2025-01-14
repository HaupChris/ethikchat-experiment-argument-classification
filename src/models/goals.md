# Ziele
Jedes der trainierten Modelle muss am Ende dazu verwendet werden können eine Wahrscheinlichkeitsverteilung der Ähnlichkeiten über alle Templates eines Argument Graphen auszugeben. Im ethikchat-argtoolkit sind im Package Classifiers bereits diverse Classifier zu finden. Diese generieren für einen gegebenen Text ein Embedding. Dieses Embedding wird mit anderen Text Embeddings abgeglichen und so das ähnlichste Argument bestimmt. 

# Modelle
Die Modelle, die in diesem Projekt trainiert werden, sind:
- Sentence Transformers
- Cross-Encoder
- Bi-Encoder https://arxiv.org/pdf/2010.08240
- Poly-Encoder
- ColBERT?
- LLM Embeddings testen (z.B. von OpenAI)

# Alternative Ansätze als Classifier
Mit LLMs sind auch andere Ansätze denkbar. z.B. über Agents, die Schritt für Schritt die Graphstruktur ausnutzen, um das beste Argument zu finden. Auch hier könnte dann eine Wahrscheinlichkeitsverteilung generiert werden. Diese würde aber eher einem One-hot-Vector entsprechen, da nur ein Argument ausgewählt wird und alle auf dem Weg dorthin verworfen werden und somit eine Wahrscheinlichkeit von 0 haben.

# Evaluation der Modelle
In ethikchat_evaluators befinden sich Evaluator Klassen für Embedding basierte Modelle. Um neue Modelle zu evaluieren, müssen diese Klassen erweitert werden.

# Nächste Schritte
- [ ] Datensatz erstellen mit Argumenten und deren Ähnlichkeiten
- [ ] Erstes Modell trainieren
- [ ] Skript zur Evaluation der Modelle schreiben
