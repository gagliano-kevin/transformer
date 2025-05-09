"""
File to test the tokenizer module.
"""

import sys
import os

# Get the absolute path to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path if it is not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from BPE_tokenizer import CustomBPETokenizer as BPE_tokenizer

# Path to the datasets
DATASETS_DIR = "../datasets"
filenames = os.listdir(DATASETS_DIR)

# Remove the file datasets_source.txt if from the filenames list
if "datasets_source.txt" in filenames:
    filenames.remove("datasets_source.txt")
# Concatenate all the files in the datasets directory into a single string
text = ""
for filename in filenames:
    with open(os.path.join(DATASETS_DIR, filename), "r", encoding="utf-8") as f:
        text += f.read()


tokenizer = BPE_tokenizer(vocab_size=260, log=True)
tokenizer.train(text)
tokenizer.tokenizer_params_dir = "./multidatasets_tokenizer_params"
tokenizer.save_model("multidatasets_tokenizer")

tokenizer = BPE_tokenizer(vocab_size=260)
tokenizer.tokenizer_params_dir = "./multidatasets_tokenizer_params"
tokenizer.load_model("multidatasets_tokenizer")


test_string = """
Mirabilia è uno studio che si occupa di Realizzazione Siti web,
Comunicazione Grafica, logo design e web Marketing con base a Palermo
per informazioni visitate il sito web www.mirabiliaweb.net o
contattate info@mirabiliaweb.net
1. DIARIO DI JONATHAN HARKER
(Stenografato).
3 maggio, Bistrita. Lasciata Monaco alle 20,35 dei primo maggio, giunto a
Vienna il mattino dopo presto; saremmo dovuti arrivare alle 6,46, ma il treno
aveva un'ora di ritardo. Stando al poco che ho potuto vederne dal treno e
percorrendone brevemente le strade, Budapest mi sembra una bellissima
città.
Non ho osato allontanarmi troppo dalla stazione, poiché, giunti in ritardo,
saremmo però ripartiti quanto più possibile in orario. Ne ho ricavato
l'impressione che, abbandonato l'Occidente, stessimo entrando nell'Oriente, e
infatti anche il più occidentale degli splendidi ponti sul Danubio, che qui è
maestosamente ampio e profondo, ci richiamava alle tradizioni della
dominazione turca.
Siamo partiti quasi in perfetto orario, e siamo giunti a buio fatto a
Klausenburg, dove ho pernottato all'albergo Royale. A pranzo, o meglio a
cena,
mi è stato servito pollo cucinato con pepe rosso, buonissimo, ma che mi ha
messo una gran sete (Ric.: farsi dare la ricetta per Mina). Ne ho parlato con il
cameriere, il quale mi ha spiegato che si chiama "paprika hendl", e che,
essendo
un piatto nazionale, avrei potuto gustarlo ovunque nei Carpazi. Ho trovato
assai
utile la mia infarinatura di tedesco; in verità, non so come potrei cavarmela
senza di essa.
Poiché a Londra avevo avuto un po' di tempo a disposizione, mi ero recato
al British Museum, nella cui biblioteca avevo compulsato libri e mappe sulla
Transilvania: mi era balenata l'idea che avrebbe potuto essermi utile qualche
informazione sul paese, visto che dovevo entrare in rapporti con un nobile del
luogo. Ho scoperto che il distretto da questi indicato si trova ai limiti orientali
del paese, proprio alla convergenza di tre stati, Transilvania, Moldavia e
"""

print(tokenizer.encode(test_string))
print(tokenizer.decode(tokenizer.encode(test_string)))