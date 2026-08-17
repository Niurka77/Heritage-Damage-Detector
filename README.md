# 🏛️ Heritage Damage Detector

Detector de daños en patrimonio histórico usando **YOLOv8**. Proyecto de investigación aplicada sobre el patrimonio arquitectónico peruano.

## 🎯 Qué hace

- Detecta **grietas, humedad, erosión y pérdida de material** en fachadas históricas
- Funciona como **aplicación web** (`app.py`) y como **app de escritorio** (`desktop_app.py`)
- Empaca como ejecutable de escritorio vía PyInstaller (`HeritageDetector.spec`)

## 📁 Estructura

```
├── app.py                  # Aplicación de detección (web/captura)
├── desktop_app.py          # Versión de escritorio
├── best.pt                 # Modelo entrenado (YOLOv8)
├── REQUIREMENTS.TXT        # Dependencias
├── INSTRUCCIONES_DE_INSTALACION.txt
├── HeritageDetector.spec   # Spec de PyInstaller
└── .env.example            # Plantilla de variables de entorno
```

## 🚀 Instalación

```bash
pip install -r REQUIREMENTS.TXT
python app.py
```

Para generar el ejecutable de escritorio:

```bash
pip install pyinstaller
pyinstaller HeritageDetector.spec
```

## 🛠️ Stack

`Python` · `YOLOv8` · `OpenCV` · `PyInstaller`

---

Investigación aplicada: **detección de fallas en patrimonio histórico con drones UAV** (Iglesia San Agustín, Arequipa). Desarrollada por **Niurka Guevara**.