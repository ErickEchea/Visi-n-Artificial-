"""
Vocabulario de Vision IA: nombres en espanol, objetos extra y grupos que CLIP desambigua.

- COCO_ES:      traduce las 80 clases de YOLO11 (se busca por nombre, no por numero).
- YOLOE_VOCAB:  objetos que COCO no tiene. YOLOE los detecta a partir del texto en ingles.
- GRUPOS_CLIP:  familias de objetos que los detectores suelen confundir entre si
                (llave de auto vs celular vs control remoto, laptop vs tablet...).
                CLIP mira el recorte y vota cual es realmente.
"""

COCO_ES = {
    "person": "Persona", "bicycle": "Bicicleta", "car": "Auto", "motorcycle": "Moto",
    "airplane": "Avión", "bus": "Autobús", "train": "Tren", "truck": "Camión",
    "boat": "Barco", "traffic light": "Semáforo", "fire hydrant": "Hidrante",
    "stop sign": "Señal de alto", "parking meter": "Parquímetro", "bench": "Banca",
    "bird": "Pájaro", "cat": "Gato", "dog": "Perro", "horse": "Caballo", "sheep": "Oveja",
    "cow": "Vaca", "elephant": "Elefante", "bear": "Oso", "zebra": "Cebra",
    "giraffe": "Jirafa", "backpack": "Mochila", "umbrella": "Paraguas", "handbag": "Bolso",
    "tie": "Corbata", "suitcase": "Maleta", "frisbee": "Frisbee", "skis": "Esquís",
    "snowboard": "Snowboard", "sports ball": "Pelota", "kite": "Cometa",
    "baseball bat": "Bate", "baseball glove": "Guante de béisbol", "skateboard": "Patineta",
    "surfboard": "Tabla de surf", "tennis racket": "Raqueta", "bottle": "Botella",
    "wine glass": "Copa", "cup": "Taza", "fork": "Tenedor", "knife": "Cuchillo",
    "spoon": "Cuchara", "bowl": "Tazón", "banana": "Plátano", "apple": "Manzana",
    "sandwich": "Sándwich", "orange": "Naranja", "broccoli": "Brócoli",
    "carrot": "Zanahoria", "hot dog": "Hot dog", "pizza": "Pizza", "donut": "Dona",
    "cake": "Pastel", "chair": "Silla", "couch": "Sofá", "potted plant": "Planta",
    "bed": "Cama", "dining table": "Mesa", "toilet": "Inodoro", "tv": "Televisión",
    "laptop": "Laptop", "mouse": "Mouse", "remote": "Control remoto",
    "keyboard": "Teclado", "cell phone": "Celular", "microwave": "Microondas",
    "oven": "Horno", "toaster": "Tostadora", "sink": "Fregadero",
    "refrigerator": "Refrigerador", "book": "Libro", "clock": "Reloj", "vase": "Florero",
    "scissors": "Tijeras", "teddy bear": "Oso de peluche", "hair drier": "Secadora",
    "toothbrush": "Cepillo de dientes",
}

# Texto que recibe YOLOE (ingles) -> etiqueta en espanol.
YOLOE_VOCAB = {
    "headphones": "Audífonos",
    "headset": "Audífonos",
    "earphones": "Audífonos",
    "wireless earbuds": "Audífonos inalámbricos",
    "earbuds charging case": "Audífonos inalámbricos",
    "car key": "Llave de auto",
    "keys": "Llaves",
    "wallet": "Cartera",
    "eyeglasses": "Lentes",
    "sunglasses": "Lentes",
    "wristwatch": "Reloj de pulsera",
    "pen": "Bolígrafo",
    "pencil": "Lápiz",
    "marker": "Plumón",
    "phone charger": "Cargador",
    "usb flash drive": "Memoria USB",
    "power bank": "Batería externa",
    "calculator": "Calculadora",
    "game controller": "Control de videojuego",
    "tablet": "Tablet",
    "lighter": "Encendedor",
    "credit card": "Tarjeta",
    "notebook": "Cuaderno",
    "speaker": "Bocina",
    "microphone": "Micrófono",
    "cap": "Gorra",
    "shoe": "Zapato",
    "cardboard box": "Caja",
    "soda can": "Lata",
    "thermos": "Termo",
    "screwdriver": "Desarmador",
    "hammer": "Martillo",
    "pliers": "Pinzas",
    "stapler": "Engrapadora",
    "computer monitor": "Monitor",
    # Clases que YOLO11 ya cubre. Se incluyen para que YOLOE no fuerce p.ej. un celular
    # como "calculadora"; sus cajas solo se usan como voto sobre la caja de YOLO11.
    "cell phone": "Celular",
    "computer mouse": "Mouse",
    "remote control": "Control remoto",
    "laptop": "Laptop",
    "keyboard": "Teclado",
    "person": "Persona",
    "bottle": "Botella",
    "cup": "Taza",
}

# Etiquetas que YOLO11 ya detecta bien: YOLOE no crea cajas nuevas con ellas.
ETIQUETAS_COCO = set(COCO_ES.values())

# Grupos de confusion. Cada etiqueta -> lista de descripciones para CLIP.
GRUPOS_CLIP = {
    "mano": {
        "Celular": ["a smartphone", "a mobile phone with a touchscreen",
                    "the back of a smartphone with camera lenses"],
        "Llave de auto": ["a car key fob with lock and unlock buttons", "a remote car key",
                          "a black flip key for a car"],
        "Llaves": ["a bunch of metal keys on a keyring", "a metal house key"],
        "Control remoto": ["a TV remote control with many buttons",
                           "an air conditioner remote control"],
        "Mouse": ["a computer mouse", "a wireless computer mouse"],
        "Audífonos": ["over-ear headphones", "a gaming headset with a microphone",
                      "wired earphones"],
        "Audífonos inalámbricos": ["wireless earbuds", "a charging case for wireless earbuds"],
        "Cartera": ["a leather wallet"],
        "Calculadora": ["a calculator"],
        "Batería externa": ["a power bank"],
        "Memoria USB": ["a usb flash drive"],
        "Reloj de pulsera": ["a wristwatch", "a smartwatch"],
        "Control de videojuego": ["a video game controller", "a gamepad"],
        "Encendedor": ["a lighter"],
        "Tarjeta": ["a credit card", "an ID card"],
        "Lentes": ["eyeglasses", "sunglasses"],
        "Bolígrafo": ["a pen", "a ballpoint pen"],
        "Lápiz": ["a pencil"],
        "Plumón": ["a marker pen"],
        "Tijeras": ["scissors"],
        "Cargador": ["a phone charger", "a power adapter"],
        "Cable": ["a cable", "a usb cable"],
        "Cepillo de dientes": ["a toothbrush"],
        "Cuchillo": ["a knife"],
    },
    "pantallas": {
        "Laptop": ["a laptop computer"],
        "Tablet": ["a tablet computer", "an iPad"],
        "Televisión": ["a television"],
        "Monitor": ["a computer monitor"],
        "Libro": ["a book"],
        "Cuaderno": ["a notebook"],
        "Teclado": ["a computer keyboard"],
    },
    "recipientes": {
        "Botella": ["a bottle", "a plastic water bottle"],
        "Taza": ["a coffee mug", "a cup"],
        "Vaso": ["a drinking glass"],
        "Copa": ["a wine glass"],
        "Lata": ["a soda can"],
        "Tazón": ["a bowl"],
        "Termo": ["a thermos"],
        "Florero": ["a vase"],
    },
    "bolsos": {
        "Mochila": ["a backpack"],
        "Bolso": ["a handbag", "a purse"],
        "Maleta": ["a suitcase"],
        "Caja": ["a cardboard box"],
        "Bolsa": ["a plastic bag", "a shopping bag"],
        "Cartera": ["an open leather wallet with cards and money"],
    },
    # YOLO llama "person" a una mano sola; CLIP distingue si se ve la persona o solo la mano.
    "persona": {
        "Persona": ["a person", "a person's face", "a person's upper body"],
        "Mano": ["a human hand", "a hand holding an object", "a close-up of fingers"],
    },
}

# Cosas que NO son el objeto. Si CLIP ve sobre todo esto:
#   - no cambia la etiqueta, y
#   - si lo confirma varias veces, la caja se considera falsa alarma y se oculta.
DISTRACTORES_CLIP = ["a human hand", "fingers", "a person's arm", "a person's face",
                     "a wooden table", "the edge of a desk", "the floor", "a carpet",
                     "a dark shadow", "a piece of paper", "an empty background", "a wall"]

# Distractores propios de un grupo (en "persona" la mano/cara SON respuestas validas).
DISTRACTORES_GRUPO = {
    "persona": ["the floor", "a carpet", "a dark shadow", "a wooden table",
                "an empty background", "a wall"],
}

# Etiqueta -> grupo al que pertenece (si aparece en varios, manda el primero)
GRUPO_DE = {}
for _g, _etqs in GRUPOS_CLIP.items():
    for _e in _etqs:
        GRUPO_DE.setdefault(_e, _g)

# "Escena": cuando ningun detector encuentra caja, CLIP revisa las cajas descartadas y el
# centro de la imagen buscando estos objetos (p.ej. audifonos, que YOLO casi no reconoce).
ESCENA_ETIQUETAS = {e: p for e, p in GRUPOS_CLIP["mano"].items()
                    if e not in ("Cable", "Cuchillo", "Cepillo de dientes")}

# Etiquetas que son "el mismo tipo de cosa": si ya hay una en pantalla, la busqueda de escena
# no agrega otra (p.ej. un recorte con la punta de la llave de auto no es un llavero nuevo).
FAMILIAS = [{"Llaves", "Llave de auto"}, {"Audífonos", "Audífonos inalámbricos"},
            {"Bolígrafo", "Lápiz", "Plumón"}, {"Cargador", "Batería externa"}]
FAMILIA_DE = {e: f for f in FAMILIAS for e in f}
# "wired earphones" confunde los cables del escritorio con audifonos: en escena solo diademas
ESCENA_ETIQUETAS["Audífonos"] = ["over-ear headphones", "a gaming headset with a microphone",
                                 "headphones with a headband and ear cushions"]
DISTRACTORES_ESCENA = DISTRACTORES_CLIP + ["a person", "a room", "a laptop computer",
                                           "a computer screen", "a cardboard box", "a chair",
                                           "a tangle of black cables", "a white charging cable",
                                           "cables on a desk"]
