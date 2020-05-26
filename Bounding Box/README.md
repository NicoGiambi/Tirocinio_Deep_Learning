# Tirocinio_Deep_Learning
area.py :
  Script per mostrare la distribuzione delle aree in un grafico
  Accetta l'arg "-c", per scegliere il nome di una classe da esaminare, se omesso mostra la          distribuzione delle aree di tutto il dataset

ratio.py :
  Script per mostrare la distribuzione delle ratio width/height e viceversa in un grafico
  Accetta l'arg "-c", per scegliere il nome di una classe da esaminare, se omesso mostra la          distribuzione delle aree di tutto il dataset.
  Dato che la proporzione tra altezza e larghezza della bounding box sarebbe ovviamente più concentrata tra zero e uno e dispersa per valori più alti, ho deciso di considerare sia width/height che l'opposto per rappresentarli entrambi nel range 0 - 1.

--------------------------------------------------------------------------------------------------

cocoapi_test.py :
  Script per estrapolare dati utili (area.txt, wh.txt, hw.txt) dal dataset Coco.
  
area.txt, hw.txt e wh.txt sono dei file in formato json contenenti i dati estrapolati con cocoapi_test.py e sevono a area.py e ratio.py per mettere a grafico i risultati.


NOTA: cocoapi_test.py viene usato solo per generare i modelli già presenti nella directory, e necessità di un path per accedere alle informazioni di Coco fatto così:
  coco/
    annotations/
       instances_train2017.json * I file sono entrambi reperibili online*
       instances_val2017.json
    images/
        train2017/
           *Scaricabili da http://images.cocodataset.org/zips/test2017.zip*
        val2017/
           *Scaricabili da http://images.cocodataset.org/zips/val2017.zip*
  
