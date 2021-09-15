##########################################################################################################################
### Fecha: 20-7-2021
### Grupo: 2
##########################################################################################################################



##########################################################################################################################
############################################### INSTALACION DE LIBRERIAS #################################################
##########################################################################################################################

# Conexion con discord
#!pip install discord
# Correr funciones asincrona en spyder
#!pip install nest_asyncio



##########################################################################################################################
################################################# IMPORTACION LIBRERIAS ##################################################
##########################################################################################################################

### BASICAS
import pandas as pd # Analisis y manipulacion de datos
import numpy # Tratamiento de matrices

### NLTK
import nltk # Procesamiento de lenguaje
nltk.download("punkt") # Tokenizer
from nltk.stem.lancaster import LancasterStemmer # Stemmer

### RED NEURONAL
import tflearn # Libreria para el desarrollo de la red neuronal
from tensorflow.python.framework import ops # Construccion de graficos

### ADICCIONALES
import json # Tratamiento de archivos .json
import random # Generador de aleatorios
import discord # Para implementar el chatbot en Discord
import nest_asyncio # Para correr funciones asincronas en spyder
nest_asyncio.apply()



##########################################################################################################################
################################################# VARIABLES GLOBALES #####################################################
##########################################################################################################################

# Cliente para conectar a discord
client = discord.Client()

# Clave para establecer conexion
key = "***********"

# Variable para habilitar o deshabilitar acceso a busqueda de hoteles
allow_access = False

# Registro de hoteles segun provincia seleccionada
hotel_registry = []



##########################################################################################################################
############################################### PREPARACION DATOS (X, Y) #################################################
##########################################################################################################################

# Cargar diccionario con intenciones y posibles respuestas
with open("diccionario.json") as file:
    info = json.load(file)
    
    
# Crear una bolsa de palabras
words = []
# Etiquetas
tags = []
# Conjunto de palabras por frase
auxX = []
# Conjunto total de etiquetas por frase
auxY = []


# Recorrer informacion del diccionario
for js in info["contenido"]:
    # Recorrer los patrone de cada intencion
    for pattern in js["patterns"]:
        # Tokenizar frases
        auxWords = nltk.word_tokenize(pattern)
        # Añadir a la bolsa de palabras
        words.extend(auxWords)
        # Variables auxiliares con palabras y etiquetas por frase
        auxX.append(auxWords)
        auxY.append(js["tag"])
        
        if js["tag"] not in tags:
            # Agrupacion de etiquetas unicas
            tags.append(js["tag"])
  
    
# Stemming
stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w!="?"]
# Ordenar palabras
words = sorted(list(set(words)))
# Ordenar etiquetas
tags = sorted(tags)


# Entrenamiento (0 si la palabra no esta, 1 si si esta)
train = []
# Tags (1 en el tag correcto, 0 en el resto)
output = []
# salida vacia inicializada en 0
emptyOutput = [0 for _ in range(len(tags))]

# Diccionario con la bolsa de palabras (indice, palabra)
for x, doc in enumerate(auxX):
    cub = []
    auxWords = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in auxWords:
            cub.append(1)
        else:
            cub.append(0)
    
    outputRow = emptyOutput[:]
    outputRow[tags.index(auxY[x])] = 1
    train.append(cub)
    output.append(outputRow)
  



##########################################################################################################################
################################################### CONSTRUCCION MODELO ##################################################
##########################################################################################################################

###### RED NEURONAL ######

# Pasar listas a arreglos
train = numpy.array(train)
output = numpy.array(output)

# Espacio de trabajpo de la red neuronal en blanco
ops.reset_default_graph()

##### Crear red neuronal
# Capa de entrada
network = tflearn.input_data(shape=[None, len(train[0])])
# Capa oculta
network = tflearn.fully_connected(network, 32)
network = tflearn.fully_connected(network, 16)
# Capa de salida
network = tflearn.fully_connected(network, len(tags), activation = "softmax")
# Probabilidad de aficacia de la prediccion
network = tflearn.regression(network)

# Modelo
mod = tflearn.DNN(network)
# Entrenar modelo
mod.fit(train, output, n_epoch=100, batch_size=16, show_metric=True)
# mod = model.load("./model/dnn.tflearn")



##########################################################################################################################
######################################################## CHATBOT #########################################################
##########################################################################################################################


########## FUNCION PARA SELECCIONAR LA PROVINCIA DE DESTINO ##########

def select_province(inp):
    
    # Referencias a variables globales
    global allow_access
    global hotel_registry
    
    
    if(inp.lower() == "ca"):
        # Cargar registro de hoteles de California
        with open("hotel_registry_ca.json") as file:
            hotel_registry = json.load(file)
            # Seleccionar la lista de hoteles
            hotel_registry = hotel_registry["registry"]
            # Habilitar acceso a registro, ya esta cargado
            allow_access = True
    
    
    elif(inp.lower() == "va"):
        # Cargar registro de hoteles de Virginia
        with open("hotel_registry_va.json") as file:
            hotel_registry = json.load(file)
            # Seleccionar la lista de hoteles
            hotel_registry = hotel_registry["registry"]
            # Habilitar acceso a registro, ya esta cargado
            allow_access = True
    
    
    elif(inp.lower() == "tx"):
        # Cargar registro de hoteles de Texas
        with open("hotel_registry_tx.json") as file:
            hotel_registry = json.load(file)
            # Seleccionar la lista de hoteles
            hotel_registry = hotel_registry["registry"]
            # Habilitar acceso a registro, ya esta cargado
            allow_access = True
    
    
    elif(inp.lower() == "fl"):
        # Cargar registro de hoteles de Florida
        with open("hotel_registry_fl.json") as file:
            hotel_registry = json.load(file)
            # Seleccionar la lista de hoteles
            hotel_registry = hotel_registry["registry"]
            # Habilitar acceso a registro, ya esta cargado
            allow_access = True
        
        
            
    
    
########## FUNCION EXTRAER LA CARACTERISTICA BUSCADA ##########

# Para evitar problemas de escriture, se solicita al usuario un numero a traves del cual se accede 
# a la etiqueta correspondiente
def find_tag(n):
    if (n == "1"):
        return "location"
    
    elif (n == "2"):
        return "service"
    
    elif (n == "3"):
        return "food"
    
    elif (n == "4"):
        return "facilities"
    
    elif (n == "5"):
        return "cleaning"
    
    elif (n == "6"):
        return "work"
    
    elif (n == "7"):
        return "price"
    
    elif (n == "8"):
        return "beach"





########## FUNCION PARA BUSCAR DOS CARACTERISTICA ##########    

# Se devuelven todos aquellos hoteles que presentan las dos etiquetas indicadas por el usuario
def search_two_features(inp):
    
    # Referencia a variables locales
    global hotel_registry
    
    # Separar la dos caracteristicas indicadas
    inp = inp.split("-")
    feature1 = inp[0]
    feature2 = inp[1]
    
    
    # Comprobar que la primera caracteristica presenta el formato adecuado
    if ((feature1.lower() != "1") and (feature1.lower() != "2") and (feature1.lower() != "3") and (feature1.lower() != "4") 
        and (feature1.lower() != "5") and (feature1.lower() != "6") and (feature1.lower() != "7") and (feature1.lower() != "8")):

        return "Incorrect character"
        
    
    # Comprobar que la segunda caracteristica presenta el formato adecuado
    elif ((feature2.lower() != "1") and (feature2.lower() != "2") and (feature2.lower() != "3") and (feature2.lower() != "4") 
        and (feature2.lower() != "5") and (feature2.lower() != "6") and (feature2.lower() != "7") and (feature2.lower() != "8")):

        return "Incorrect character"
    
    
    else:
        # Obtener las etiquetas seleccionadas
        tag1 = find_tag(feature1.lower())
        tag2 = find_tag(feature2.lower())

        # Obtener palabras asociadas a dichas etiquetas
        # TAG 1
        features_tofind_tag1 = []
        for f in tag_features:
            if f["tag"] == tag1:
                features_tofind_tag1 = f["patterns"]
                
        # TAG 2
        features_tofind_tag2 = []
        for f in tag_features:
            if f["tag"] == tag2:
                features_tofind_tag2 = f["patterns"]
                
                
        # Obtener lista de hoteles que tienen las dos etiquetas
        hotels = pd.DataFrame(columns = ['Hotel', 'Loc', 'Percent'])
        for i in range(len(hotel_registry)):
            h_registry = hotel_registry[i]
            dic_features = h_registry['Features']
            
            # Contadores para extraer el numero de caracteristicas asociadas a cada etiqueta
            count_tag1 = 0
            count_tag2 = 0
            
            for fea in dic_features:
                if (fea in features_tofind_tag1):
                    count_tag1 += 1
                
                elif (fea in features_tofind_tag2):
                    count_tag2 += 1
            
            # Ambos contadores presentan al menos una feature, se añade hotel 
            if (count_tag1 !=0 and count_tag2 != 0):
                hotels = hotels.append({'Hotel':h_registry["Hotel"], 'Loc':h_registry["Location"], 
                               'Percent':((dic_features[fea]/h_registry["Reviews"]) * 100)}, ignore_index=True)
    
        
        # Comprobar si se ha detectado algun hotel con ambas etiquetas
        if len(hotels.index)==0:
            return "No hotel found"
            
            
        else:
            string = "RECOMMENDATIONS: \n     "
            
            for i in range(len(hotels)):
                h = hotels.iloc[i]
                string = string + h['Hotel'] + ", " +  h["Loc"] + "\n     "
                
            return string


    
    
    
########## FUNCION ENCONTRAR MEJOR HOTEL SEGUN NECESIDADES ##########

### Cargar caracteristicas a buscar en los hoteles
with open("tag_features.json") as file:
    tag_features = json.load(file)
    
### Seleccionar lista de caracteristicas
tag_features = tag_features["features"]


def search_suitHotel(inp):
    
    # Referencia a variables globales
    global hotel_registry

    # Obtener la etiqueta seleccionada
    tag = find_tag(inp.lower())
    
    # Obtener palabras asociadas a dicha etiqueta
    features_tofind = []
    for f in tag_features:
        if f["tag"] == tag:
            features_tofind = f["patterns"]
            
    # Lista de hoteles que contienen la etiqueta
    hotels = pd.DataFrame(columns = ['Hotel', 'Loc', 'Percent'])
    for i in range(len(hotel_registry)):
        for fea in features_tofind:
            h_registry = hotel_registry[i]
            dic_features = h_registry['Features']
            if(fea in dic_features):
                hotels = hotels.append({'Hotel':h_registry["Hotel"], 'Loc':h_registry["Location"], 
                               'Percent':((dic_features[fea]/h_registry["Reviews"]) * 100)}, ignore_index=True)
    
    
    # Comprobar si se ha encontrado algun hotel con la etiqueta indicada
    if len(hotels.index)==0:
        return "No hotel found"
        
        
    elif len(hotels.index)<=3:
        string = "RECOMMENDATIONS: \n     "
        
        for i in range(len(hotels)):
            h = hotels.iloc[i]
            string = string + h['Hotel'] + ", " +  h["Loc"] + "\n     "
            
        return string
    
    # Para evitar proporcionar una lista muy extensa, en caso de existir mas de tres hoteles que presenten la etiqueta, 
    # se devuelven unicamente los tres que mayor puntuacion presenten en dicha etiqueta
    else:
        # Oredenar hoteles por mayor porcentaje 
        by_percent = hotels.sort_values('Percent',ascending=False)
        # Mostrar hoteles con mayor valoracion en dicha feature 
        string = "RECOMMENDATIONS: \n     " 
        
        for i in [0,1,2]:
            row = by_percent.iloc[i]
            string = string + row['Hotel'] + ", " + row['Loc'] + "\n     "
            
        return string
        




########## FUNCION CHATBOT ##########
        
def mainChatBot(msg):
    
    # Referencia a variables globales
    global allow_access
    
    i = msg

    if (i == 'finish'):
        for t in info["contenido"]:
            if t["tag"] == "farewell":
                answer = t["answers"]

        return random.choice(answer)
        
    
    else:
        # Obtener la etiqueta detectada por el modelo segun el texto enviado
        cub = [0 for _ in range(len(words))]
        processInput = nltk.word_tokenize(i)
        processInput = [stemmer.stem(word.lower()) for word in processInput]
        for w in processInput:
            for i, word in enumerate(words):
                if word == w:
                    cub[i] = 1
        result = mod.predict([numpy.array(cub)])
        indexResult = numpy.argmax(result)
        
        # Comprobar que la etiqueta supere un umbral de confianza
        if numpy.max(result)<0.4:
            tag = "noanswer"
        else:
            tag = tags[indexResult]
            
        # Si la etiqueta ddetectada no es la de busqueda, actua como un chatbot normal
        if tag != "search":
            for t in info["contenido"]:
                if t["tag"] == tag:
                    answer = t["answers"]

            return random.choice(answer)
        
        # En caso contrario, entra en la funcionalidad adicional de recomendador
        else:
            for t in info["contenido"]:
                if t["tag"] == tag:
                    answer = t["answers"]
            
            string = "Select a vacation destination: \n     - CA-California \n     - VA-Virginia \n     - TX-Texas \n     - FL-Florida"
            return string

        


        
########## FUNCION PARA CONECTAR EL CHARBOT CHATBOT CON DISCORD ##########
        
def connect_toDiscord():
    # Referencia a variables globales
    global client
    global key
    global allow_access
    
    # Evento, activado al recibir mensaje de discord
    @client.event
    async def on_message(msg):
        
        # Comprobar que el mensaje recibido no sea la contestacion del propio bot
        if msg.author == client.user:
            return
        
        else:
            
            if (msg.content.find("ca")!=-1 or msg.content.find("va")!=-1 or msg.content.find("tx")!=-1 or msg.content.find("fl")!=-1):
                select_province(msg.content.strip())
                result = "AVAILABLE HOTEL FEATURES: \n     - 1) Location \n     - 2) Service \n     - 3) Food \n     - 4) Facilities \n     - 5) Cleaning \n     - 6) Work \n     - 7) Price \n     - 8) Beach \nEnter a numbre between 1 and 8 or two numbers separated by a hyphen"
               
            elif ((len(msg.content.replace(" ", ""))==1) and ((msg.content.replace(" ", "") == "1") or (msg.content.replace(" ", "") == "2") or (msg.content.replace(" ", "") == "3") or (msg.content.replace(" ", "") == "4")
                                                              or (msg.content.replace(" ", "") == "5") or (msg.content.replace(" ", "") == "6") or (msg.content.replace(" ", "") == "7") or (msg.content.replace(" ", "") == "8"))
                                                         and (allow_access== True)):
                result = search_suitHotel(msg.content.replace(" ", ""))
                
            elif ((len(msg.content.replace(" ", ""))==3) and (msg.content.replace(" ", "")[1] == "-")):
                  result = search_two_features(msg.content.replace(" ", ""))
               
            else:    
                result = mainChatBot(msg.content)
    
            await msg.channel.send(result)
    
    # Activar el cliente
    client.run(key)
 
    
 
    
########## EJECUTAR CHATBOT RECOMENDADOR ##########   
connect_toDiscord()
