# -*- coding: utf-8 -*-
"""
detection_gravity.py - Módulo de Clasificación de Gravedad y Priorización
Heritage Damage Detector v6.5
Universidad Católica de Santa María - Arequipa, Perú

Clasifica cada detección individual según su área y tipo de patología.
Genera reportes de priorización para intervención arquitectónica.
"""

# =============================================================================
# CONFIGURACIÓN DE UMBRALES POR PATOLOGÍA (en cm²)
# =============================================================================
# Estos umbrales están calibrados para el Sillar Volcánico de Arequipa
# y pueden ajustarse según criterio del arquitecto supervisor.

GRAVITY_THRESHOLDS = {
    "crack": {
        "leve":     (0, 5),         # Fisuras capilares superficiales
        "moderada": (5, 30),        # Fisuras visibles, sin compromiso estructural
        "severa":   (30, 100),      # Grietas que afectan la estética y pueden profundizarse
        "critica":  (100, 999999)   # Grietas estructurales, intervención urgente
    },
    "humidity": {
        "leve":     (0, 50),        # Manchas superficiales
        "moderada": (50, 200),      # Humedad localizada, posible filtración
        "severa":   (200, 500),     # Humedad extendida, riesgo de deterioro
        "critica":  (500, 999999)   # Infiltración masiva, intervención urgente
    },
    "spalling": {
        "leve":     (0, 20),        # Pérdida superficial de material
        "moderada": (20, 100),      # Desprendimiento localizado
        "severa":   (100, 300),     # Desprendimiento extenso
        "critica":  (300, 999999)   # Pérdida estructural de sillar
    }
}

# =============================================================================
# COLORES POR NIVEL DE GRAVEDAD (HEX)
# =============================================================================
# Paleta vibrante y clara (evita colores oscuros según preferencias del usuario)

GRAVITY_COLORS = {
    "leve":     "#10B981",   # Verde esmeralda - daño mínimo
    "moderada": "#F59E0B",   # Ámbar - atención
    "severa":   "#F97316",   # Naranja - intervención próxima
    "critica":  "#DC2626"    # Rojo - intervención inmediata
}

# =============================================================================
# SCORES DE PRIORIDAD (mayor = más urgente)
# =============================================================================

GRAVITY_SCORE = {
    "leve":     1,
    "moderada": 2,
    "severa":   3,
    "critica":  4
}

# =============================================================================
# ETIQUETAS LEGIBLES
# =============================================================================

GRAVITY_LABELS = {
    "leve":     "🟢 Leve",
    "moderada": "🟡 Moderada",
    "severa":   "🟠 Severa",
    "critica":  "🔴 CRÍTICA"
}

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def classify_gravity(class_name: str, area_cm2: float) -> str:
    """
    Clasifica una detección individual según su tipo y área.
    
    Args:
        class_name: Nombre de la clase ("crack", "humidity", "spalling")
        area_cm2: Área de la detección en cm²
    
    Returns:
        Nivel de gravedad: "leve", "moderada", "severa" o "critica"
    """
    thresholds = GRAVITY_THRESHOLDS.get(class_name, GRAVITY_THRESHOLDS["crack"])
    
    for level, (min_a, max_a) in thresholds.items():
        if min_a <= area_cm2 < max_a:
            return level
    
    return "leve"


def enrich_detections_with_gravity(detections: list) -> list:
    """
    Añade información de gravedad a cada detección y las ordena por prioridad.
    
    Añade los siguientes campos a cada detección:
        - gravedad: nivel ("leve", "moderada", "severa", "critica")
        - gravedad_score: puntuación numérica (1-4)
        - gravedad_color: color HEX para visualización
        - gravedad_label: etiqueta legible con emoji
    
    Args:
        detections: Lista de detecciones con al menos "Clase" y "Area_cm2"
    
    Returns:
        Lista enriquecida, ordenada de mayor a menor gravedad
    """
    for d in detections:
        class_name = d.get("Clase", "crack")
        area_cm2 = d.get("Area_cm2", 0)
        
        gravity = classify_gravity(class_name, area_cm2)
        
        d["gravedad"] = gravity
        d["gravedad_score"] = GRAVITY_SCORE[gravity]
        d["gravedad_color"] = GRAVITY_COLORS[gravity]
        d["gravedad_label"] = GRAVITY_LABELS[gravity]
    
    # Ordenar: primero las más graves
    return sorted(detections, key=lambda x: x["gravedad_score"], reverse=True)


def generate_priority_report(detections: list, filename: str) -> dict:
    """
    Genera un reporte completo de priorización para intervención arquitectónica.
    
    Args:
        detections: Lista de detecciones enriquecidas con gravedad
        filename: Nombre del archivo inspeccionado
    
    Returns:
        Diccionario con:
            - filename: nombre del archivo
            - total_priority_score: score total de prioridad
            - urgency: nivel de urgencia global ("CRÍTICA", "ALTA", "MEDIA", "BAJA")
            - recommendation: texto con recomendación de intervención
            - criticas_count: cantidad de detecciones críticas
            - severas_count: cantidad de detecciones severas
            - moderadas_count: cantidad de detecciones moderadas
            - leves_count: cantidad de detecciones leves
            - top_3_detections: las 3 detecciones más graves
            - all_detections: todas las detecciones ordenadas por gravedad
            - area_by_gravity: área total agrupada por nivel de gravedad
    """
    enriched = enrich_detections_with_gravity(detections)
    
    # Conteos por nivel
    criticas = [d for d in enriched if d.get("gravedad") == "critica"]
    severas = [d for d in enriched if d.get("gravedad") == "severa"]
    moderadas = [d for d in enriched if d.get("gravedad") == "moderada"]
    leves = [d for d in enriched if d.get("gravedad") == "leve"]
    
    # Score total de prioridad
    total_priority_score = sum(d.get("gravedad_score", 1) for d in enriched)
    
    # Área agrupada por gravedad
    area_by_gravity = {
        "critica": sum(d.get("Area_cm2", 0) for d in criticas),
        "severa": sum(d.get("Area_cm2", 0) for d in severas),
        "moderada": sum(d.get("Area_cm2", 0) for d in moderadas),
        "leve": sum(d.get("Area_cm2", 0) for d in leves)
    }
    
    # Determinar urgencia global y recomendación
    if len(criticas) > 0:
        recommendation = "🚨 INTERVENCIÓN INMEDIATA requerida"
        urgency = "CRÍTICA"
    elif len(severas) > 2:
        recommendation = "⚠️ Programar intervención en < 30 días"
        urgency = "ALTA"
    elif total_priority_score > 15:
        recommendation = "📋 Programar intervención en < 90 días"
        urgency = "MEDIA"
    else:
        recommendation = "✅ Monitoreo preventivo"
        urgency = "BAJA"
    
    return {
        "filename": filename,
        "total_priority_score": total_priority_score,
        "urgency": urgency,
        "recommendation": recommendation,
        "criticas_count": len(criticas),
        "severas_count": len(severas),
        "moderadas_count": len(moderadas),
        "leves_count": len(leves),
        "top_3_detections": enriched[:3],
        "all_detections": enriched,
        "area_by_gravity": area_by_gravity
    }


def get_gravity_summary(detections: list) -> dict:
    """
    Genera un resumen rápido de la distribución de gravedades.
    Útil para mostrar en dashboards o tooltips.
    """
    summary = {"leve": 0, "moderada": 0, "severa": 0, "critica": 0}
    for d in detections:
        gravity = d.get("gravedad", "leve")
        if gravity in summary:
            summary[gravity] += 1
    return summary


def should_alert_by_gravity(detections: list) -> list:
    """
    Evalúa si se deben generar alertas basadas en gravedades individuales.
    Complementa el sistema de alertas globales.
    
    Returns:
        Lista de alertas generadas (puede estar vacía)
    """
    from datetime import datetime
    
    alerts = []
    enriched = enrich_detections_with_gravity(detections)
    
    criticas = [d for d in enriched if d.get("gravedad") == "critica"]
    severas = [d for d in enriched if d.get("gravedad") == "severa"]
    
    # Alerta si hay al menos 1 detección crítica
    if criticas:
        alerts.append({
            "level": "CRÍTICA",
            "type": "Detecciones críticas individuales",
            "description": f"{len(criticas)} detección(es) requieren intervención inmediata",
            "timestamp": datetime.now().isoformat()
        })
    
    # Alerta si hay muchas detecciones severas
    if len(severas) >= 3:
        alerts.append({
            "level": "ALTA",
            "type": "Acumulación de daños severos",
            "description": f"{len(severas)} detecciones severas detectadas",
            "timestamp": datetime.now().isoformat()
        })
    
    return alerts


# =============================================================================
# UTILIDADES DE VISUALIZACIÓN
# =============================================================================

def hex_to_rgb_tuple(hex_color: str) -> tuple:
    """Convierte color HEX a tupla RGB (0-255)"""
    hex_clean = hex_color.lstrip('#')
    return tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgb_normalized(hex_color: str) -> tuple:
    """Convierte color HEX a tupla RGB normalizada (0.0-1.0) para matplotlib"""
    r, g, b = hex_to_rgb_tuple(hex_color)
    return (r / 255.0, g / 255.0, b / 255.0)


# =============================================================================
# EJEMPLO DE USO (solo para pruebas)
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de detecciones simuladas
    test_detections = [
        {"Clase": "crack", "Area_cm2": 2.5, "Confianza": 0.85},
        {"Clase": "crack", "Area_cm2": 45.0, "Confianza": 0.92},
        {"Clase": "humidity", "Area_cm2": 150.0, "Confianza": 0.78},
        {"Clase": "spalling", "Area_cm2": 350.0, "Confianza": 0.88},
        {"Clase": "crack", "Area_cm2": 120.0, "Confianza": 0.95},
    ]
    
    print("=" * 60)
    print("PRUEBA DE CLASIFICACIÓN DE GRAVEDAD")
    print("=" * 60)
    
    report = generate_priority_report(test_detections, "test_image.jpg")
    
    print(f"\n📁 Archivo: {report['filename']}")
    print(f"🎯 Urgencia: {report['urgency']}")
    print(f"💡 Recomendación: {report['recommendation']}")
    print(f"📊 Score total: {report['total_priority_score']}")
    
    print(f"\n📈 Distribución:")
    print(f"   🔴 Críticas: {report['criticas_count']}")
    print(f"   🟠 Severas:  {report['severas_count']}")
    print(f"   🟡 Moderadas: {report['moderadas_count']}")
    print(f"   🟢 Leves:    {report['leves_count']}")
    
    print(f"\n🏆 Top 3 detecciones más graves:")
    for i, d in enumerate(report['top_3_detections'], 1):
        print(f"   {i}. {d['Clase']} - {d['gravedad_label']} - {d['Area_cm2']:.2f} cm²")
    
    print(f"\n📏 Área por nivel de gravedad (cm²):")
    for level, area in report['area_by_gravity'].items():
        if area > 0:
            print(f"   {GRAVITY_LABELS[level]}: {area:.2f} cm²")