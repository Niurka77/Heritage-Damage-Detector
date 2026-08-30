# alerts_system.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import json
import os
from pathlib import Path
from datetime import datetime

# ✅ CORRECCIÓN: Usar carpeta del usuario para logs (evita PermissionError en Program Files)
log_dir = Path.home() / "Documents" / "HeritageDetector"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "alertas.log"

# Configurar logging con ruta segura
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # ✅ Ahora es escribible
        logging.StreamHandler()
    ]
)

class AlertSystem:
    def __init__(self, thresholds: dict = None):
        """
        thresholds: dict con umbrales personalizables
        Ej: {
            "max_damage_area_m2": 0.5,
            "min_new_cracks": 10,
            "growth_percentage_threshold": 20
        }
        """
        self.thresholds = thresholds or {
            "max_damage_area_m2": 0.5,
            "min_new_cracks": 10,
            "growth_percentage_threshold": 20,
            "min_avg_confidence": 70.0
        }

    def evaluate_current_result(self, result, last_history=None):
        """
        Evalúa un resultado actual contra umbrales o historial.
        Retorna lista de alertas.
        """
        alerts = []

        # Alerta por área total de daño
        if result.get("total_area_m2", 0) > self.thresholds["max_damage_area_m2"]:
            alerts.append({
                "level": "ALTA",
                "type": "Área de daño excesiva",
                "description": f"Área total: {result['total_area_m2']:.4f} m²",
                "timestamp": datetime.now().isoformat()
            })

        # Alerta por cantidad de grietas
        if result["class_counts"].get("crack", 0) > self.thresholds["min_new_cracks"]:
            alerts.append({
                "level": "MEDIA",
                "type": "Alta cantidad de grietas",
                "description": f"Grietas detectadas: {result['class_counts']['crack']}",
                "timestamp": datetime.now().isoformat()
            })

        # Alerta por confianza baja
        if result.get("avg_confidence", 0) < self.thresholds["min_avg_confidence"]:
            alerts.append({
                "level": "MEDIA",
                "type": "Baja confianza promedio",
                "description": f"Confianza: {result['avg_confidence']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })

        # Comparación histórica (si hay historial)
        if last_history:
            last_area = last_history.get("total_area_m2", 0)
            growth = ((result["total_area_m2"] - last_area) / max(last_area, 0.001)) * 100
            if growth > self.thresholds["growth_percentage_threshold"]:
                alerts.append({
                    "level": "ALTA",
                    "type": "Crecimiento significativo de daño",
                    "description": f"Crecimiento: {growth:.1f}%",
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts

    def log_alerts(self, alerts):
        """Registra alertas en archivo de log"""
        for alert in alerts:
            logging.warning(f"[{alert['level']}] {alert['type']}: {alert['description']}")

    def send_email_alerts(self, alerts, email_config):
        """
        Opcional: Enviar alertas por correo.
        email_config: {
            "smtp_server": "...",
            "port": 587,
            "username": "...",
            "password": "...",
            "to_emails": ["admin@ucsm.edu.pe"]
        }
        """
        if not alerts:
            return
        
        subject = f"[ALERTA HERITAGE] {len(alerts)} alertas detectadas"
        body = f"Se han generado {len(alerts)} alertas:\n\n"
        for alert in alerts:
            body += f"- [{alert['level']}] {alert['type']}: {alert['description']}\n"

        msg = MIMEMultipart()
        msg["From"] = email_config["username"]
        msg["To"] = ", ".join(email_config["to_emails"])
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(email_config["smtp_server"], email_config["port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.sendmail(email_config["username"], email_config["to_emails"], msg.as_string())
            server.quit()
            logging.info("Correo de alerta enviado.")
        except Exception as e:
            logging.error(f"Error enviando correo: {e}")