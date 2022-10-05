Installationsvorgang (Mit einer unveränderten neuen Windows-Installation getestet)
Achtung: Stellen sie sicher, dass etwa 4 GB an Speicherplatz vorhanden sind

Zur Installation der benötigten Dateien brauchen sie git. Installieren sie git.

Um alles zu installieren, können sie entweder die Dateien in einem neuen Pfad installieren, oder über git die Dateien herunterladen:
	git clone https://github.com/manuel-gross/tweet_customer_support_lfqa.git

Extrahieren sie faiss_document_store.db aus dem Zip-Ordner und platzieren sie die Datei in den gleichen Ordner wie app.py; config; index

Installieren sie Python 3.10 (oder 3.7 wenn sie die 3.7 Version verwenden wollen)

Es kann sein, dass beim Versuch, Haystack ganz zu installieren, es zu Problemen aufgrund von speziellen Dateien kommen kann.
Um dies zu lösen, gehen sie in ihren Git-Installationsordner und öffnen sie die Konfigurationsdatei unter \Git\etc\gitconfig
Fügen sie unter [core] folgenden Text hinzu:
	longpaths = true

Installieren sie die Haystack-Bibliothek:

Öffnen sie ein neues Konsolenfenster und gehen sie auf einen neuen Pfad ihrer Wahl
Geben sie folgendes in das Konsolenfenster ein:
	git clone https://github.com/deepset-ai/haystack.git
	cd haystack
	pip install --upgrade pip
	pip install -e .[sql,only-faiss,only-milvus1,weaviate,graphdb,crawler,preprocessing,ocr,onnx,dev]

Falls es Probleme gibt, versuchen sie zuerst die Standard-Installation von Haystack vor der gesamten Installation durchzuführen, indem sie in ihr Konsolenfenster folgendes eingeben:
	pip install farm-haystack

Um das Programm zu starten, führen sie app.py (oder app_ver_3.7.py falls sie auf Python 3.7 sind) aus
(app_evaluation.py kann von beiden Versionen gestartet werden)
