# -*- coding: utf-8 -*-
"""
desktop_app.py - Heritage Damage Detector v6.5 DESKTOP EDITION
Universidad Católica de Santa María - Arequipa, Perú
Aplicación de escritorio nativa con CustomTkinter + Supabase
Motor: Tiling + NMS + Filtros Geométricos + Anti-sombras
v6.5: Hover interactivo + Clasificación de gravedad + Priorización
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from PIL import Image, ImageTk
from alerts_system import AlertSystem
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import os
import sys
import json
import re
import time
import torch
import requests
import threading
from pathlib import Path
from dotenv import load_dotenv

def _cargar_config():
    """Carga .env desde el directorio empaquetado (PyInstaller) o el directorio actual."""
    try:
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            return
    except Exception:
        pass
    load_dotenv()

_cargar_config()

# Intentar importar detection_gravity (si existe)
try:
    from detection_gravity import enrich_detections_with_gravity, generate_priority_report, GRAVITY_COLORS
    GRAVITY_MODULE_AVAILABLE = True
except ImportError:
    GRAVITY_MODULE_AVAILABLE = False
    # Valores por defecto si el módulo no existe aún
    GRAVITY_COLORS = {
        "leve": "#10B981",
        "moderada": "#F59E0B",
        "severa": "#F97316",
        "critica": "#DC2626"
    }
    
    def enrich_detections_with_gravity(detections):
        """Fallback: añade gravedad básica si no hay módulo"""
        for d in detections:
            area = d.get("Area_cm2", 0)
            if area > 100:
                gravity = "critica"
            elif area > 30:
                gravity = "severa"
            elif area > 5:
                gravity = "moderada"
            else:
                gravity = "leve"
            d["gravedad"] = gravity
            d["gravedad_score"] = {"leve": 1, "moderada": 2, "severa": 3, "critica": 4}[gravity]
            d["gravedad_color"] = GRAVITY_COLORS[gravity]
            d["gravedad_label"] = {"leve": "🟢 Leve", "moderada": "🟡 Moderada", "severa": "🟠 Severa", "critica": "🔴 CRÍTICA"}[gravity]
        return sorted(detections, key=lambda x: x["gravedad_score"], reverse=True)
    
    def generate_priority_report(detections, filename):
        """Fallback: genera reporte básico"""
        enriched = enrich_detections_with_gravity(detections)
        criticas = [d for d in enriched if d.get("gravedad") == "critica"]
        severas = [d for d in enriched if d.get("gravedad") == "severa"]
        total_score = sum(d.get("gravedad_score", 1) for d in enriched)
        
        if len(criticas) > 0:
            recommendation = "🚨 INTERVENCIÓN INMEDIATA requerida"
            urgency = "CRÍTICA"
        elif len(severas) > 2:
            recommendation = "⚠️ Programar intervención en < 30 días"
            urgency = "ALTA"
        elif total_score > 15:
            recommendation = "📋 Programar intervención en < 90 días"
            urgency = "MEDIA"
        else:
            recommendation = "✅ Monitoreo preventivo"
            urgency = "BAJA"
        
        return {
            "filename": filename,
            "total_priority_score": total_score,
            "urgency": urgency,
            "recommendation": recommendation,
            "criticas_count": len(criticas),
            "severas_count": len(severas),
            "top_3_detections": enriched[:3],
            "all_detections": enriched
        }

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

MODEL_PATH = "best.pt"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

COLORS = {
    "bg_primary":      "#F8FAFC",
    "bg_secondary":    "#FFFFFF",
    "bg_surface":      "#F1F5F9",
    "bg_panel":        "#F1F5F9",
    "bg_card":         "#FFFFFF",
    "primary":         "#1D4ED8",
    "primary_hover":   "#1E40AF",
    "accent_gold":     "#D97706",
    "success":         "#059669",
    "success_light":   "#D1FAE5",
    "crack":           "#DC2626",
    "humidity":        "#0891B2",
    "spalling":        "#B45309",
    "text_primary":    "#1E293B",
    "text_secondary":  "#475569",
    "text_muted":      "#64748B",
    "border":          "#E2E8F0",
    "danger":          "#DC2626",
}

FONT_FAMILY = "Segoe UI"

# =============================================================================
# CLIENTE SUPABASE
# =============================================================================

class SupabaseClient:
    """Cliente REST para Supabase"""
    
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        # Las nuevas publishable/secret keys (sb_publishable_/sb_secret_) se
        # envian SOLO en el header 'apikey'. El header 'Authorization: Bearer'
        # queda reservado para JWTs (legacy anon/service_role).
        is_jwt = not key.startswith("sb_publishable_") and not key.startswith("sb_secret_")
        self.headers = {
            "apikey": key,
            "Content-Type": "application/json",
        }
        if is_jwt:
            self.headers["Authorization"] = f"Bearer {key}"
    
    def ping(self) -> tuple:
        try:
            r = requests.get(
                f"{self.url}/rest/v1/inspection_results",
                headers={**self.headers, "Prefer": "count=exact"},
                params={"select": "id", "limit": "1"},
                timeout=8,
            )
            if 200 <= r.status_code < 300:
                return True, "Conectado a Supabase"
            return False, f"Error HTTP {r.status_code} al conectar con Supabase"
        except Exception as e:
            return False, "Sin conexión con Supabase"
    
    def insert(self, table: str, data) -> dict:
        try:
            headers = {**self.headers, "Prefer": "return=representation"}
            r = requests.post(
                f"{self.url}/rest/v1/{table}",
                headers=headers,
                json=data,
                timeout=15,
            )
            if r.status_code in (200, 201):
                return {"data": r.json(), "error": None, "success": True, "status": r.status_code}
            return {"data": None, "error": f"Error HTTP {r.status_code} en Supabase", "success": False, "status": r.status_code}
        except Exception as e:
            return {"data": None, "error": "Error de conexión con Supabase", "success": False, "status": 0}
    
    def select(self, table: str, query: str = "*", order: str = None, 
               limit: int = None, filters: dict = None) -> dict:
        try:
            params = {"select": query}
            if order:
                params["order"] = order
            if limit:
                params["limit"] = limit
            if filters:
                params.update(filters)
            
            r = requests.get(
                f"{self.url}/rest/v1/{table}",
                headers=self.headers,
                params=params,
                timeout=30,
            )
            if 200 <= r.status_code < 300:
                return {"data": r.json(), "error": None, "success": True, "status": r.status_code}
            return {"data": None, "error": f"Error HTTP {r.status_code} en Supabase", "success": False, "status": r.status_code}
        except Exception as e:
            return {"data": None, "error": "Error de conexión con Supabase", "success": False, "status": 0}
    
    def delete_inspection(self, inspection_id: int) -> dict:
        try:
            r = requests.delete(
                f"{self.url}/rest/v1/inspection_results",
                headers=self.headers,
                params={"id": f"eq.{inspection_id}"},
                timeout=10,
            )
            if 200 <= r.status_code < 300:
                return {"success": True, "status": r.status_code}
            return {"success": False, "status": r.status_code, "error": f"Error HTTP {r.status_code} en Supabase"}
        except Exception as e:
            return {"success": False, "error": "Error de conexión con Supabase"}
    
    def check_duplicate_files(self, filenames: list) -> dict:
        if not filenames:
            return {"duplicates": [], "new_files": []}
        
        try:
            safe_pattern = re.compile(r"^[\w.\- áéíóúÁÉÍÓÚñÑüÜ]+$")
            filenames_clean = []
            for f in filenames:
                if safe_pattern.match(f):
                    filenames_clean.append(f.replace("'", "''"))
            if not filenames_clean:
                return {"duplicates": [], "new_files": filenames}

            filter_value = f"in.({','.join(filenames_clean)})"
            
            result = self.select(
                "inspection_results",
                "filename",
                filters={"filename": filter_value}
            )
            
            if result["success"] and result["data"]:
                existing_files = {row["filename"] for row in result["data"]}
                duplicates = [f for f in filenames if f in existing_files]
                new_files = [f for f in filenames if f not in existing_files]
                return {"duplicates": duplicates, "new_files": new_files}
            
            return {"duplicates": [], "new_files": filenames}
        except Exception as e:
            print(f"Error verificando duplicados: {e}")
            return {"duplicates": [], "new_files": filenames}
    
    def save_inspection(self, result, filename):
        inspection_id = None
        try:
            inspection_payload = {
                "filename": str(filename),
                "crack_count": int(result["class_counts"].get("crack", 0)),
                "humidity_count": int(result["class_counts"].get("humidity", 0)),
                "spalling_count": int(result["class_counts"].get("spalling", 0)),
                "total_detections": int(result["n_detections"]),
                "total_area_cm2": float(result["total_area_cm2"]),
                "total_area_m2": float(result["total_area_m2"]),
                "avg_confidence": float(result.get("avg_confidence", 0)),
            }
            
            resp = self.insert("inspection_results", inspection_payload)
            
            if not resp["success"]:
                return False, resp.get("error", "Error desconocido")
            
            if resp.get("data") and len(resp["data"]) > 0:
                inspection_id = resp["data"][0].get("id")
            
            if not inspection_id:
                return False, "No se recibió ID"
            
            detections = result.get("detections", [])
            if detections:
                details_payload = []
                for d in detections:
                    conf_val = float(d["Confianza"])
                    if conf_val > 1.0:
                        conf_val = conf_val / 100.0
                    
                    details_payload.append({
                        "inspection_id": int(inspection_id),
                        "class_name": str(d["Clase"]),
                        "confidence": float(conf_val),
                        "x1": int(d["x1"]),
                        "y1": int(d["y1"]),
                        "x2": int(d["x2"]),
                        "y2": int(d["y2"]),
                        "width_px": int(d["Ancho_px"]),
                        "height_px": int(d["Alto_px"]),
                        "area_cm2": float(d["Area_cm2"]),
                        "area_m2": float(d["Area_m2"]),
                        "gravity_level": d.get("gravedad", "leve"),
                        "gravity_score": d.get("gravedad_score", 1),
                    })
                
                if details_payload:
                    resp2 = self.insert("detection_details", details_payload)
                    if not resp2["success"]:
                        if inspection_id:
                            self.delete_inspection(inspection_id)
                        return False, f"Detalles: {resp2.get('error')} - Rollback ejecutado"
            
            return True, inspection_id
        except Exception as e:
            if inspection_id:
                self.delete_inspection(inspection_id)
            return False, str(e)
    
    def load_historical_data(self, limit=100):
        result = self.select(
            "inspection_results",
            "*",
            order="created_at.desc",
            limit=limit
        )
        if result["success"]:
            return result["data"]
        return []
    
    def load_detection_details(self, inspection_ids):
        if not inspection_ids:
            return {}
        
        ids_str = ", ".join(str(int(id)) for id in inspection_ids if id is not None)
        if not ids_str:
            return {}
        
        result = self.select(
            "detection_details",
            "inspection_id, class_name, area_m2, area_cm2, confidence, x1, y1, x2, y2, width_px, height_px, gravity_level, gravity_score",
            filters={"inspection_id": f"in.({ids_str})"},
            limit=10000
        )
        
        if not result["success"] or not result["data"]:
            return {}
        
        details_by_inspection = {}
        for detail in result["data"]:
            insp_id = detail.get("inspection_id")
            if insp_id not in details_by_inspection:
                details_by_inspection[insp_id] = []
            details_by_inspection[insp_id].append(detail)
        
        return details_by_inspection
    
    def get_stats(self):
        data = self.select("inspection_results", "*", limit=10000)
        if not data["success"] or not data["data"]:
            return {
                "total_inspections": 0,
                "total_detections": 0,
                "total_area_m2": 0,
                "avg_confidence": 0,
                "total_cracks": 0,
                "total_humidity": 0,
                "total_spalling": 0,
            }
        
        rows = data["data"]
        return {
            "total_inspections": len(rows),
            "total_detections": sum(r.get("total_detections", 0) for r in rows),
            "total_area_m2": sum(float(r.get("total_area_m2", 0)) for r in rows),
            "avg_confidence": sum(float(r.get("avg_confidence", 0)) for r in rows) / max(len(rows), 1),
            "total_cracks": sum(r.get("crack_count", 0) for r in rows),
            "total_humidity": sum(r.get("humidity_count", 0) for r in rows),
            "total_spalling": sum(r.get("spalling_count", 0) for r in rows),
        }

# =============================================================================
# MOTOR DE PROCESAMIENTO
# =============================================================================

def filtrar_falsos_positivos(detections, image_shape, umbrales_clase):
    """Filtro geométrico + anti-sombras"""
    filtered_detections = []
    height, width = image_shape[:2]
    image_area = width * height
    
    for d in detections:
        cls_name = d["Clase"]
        conf = d["Confianza"]
        if conf > 1.0:
            conf = conf / 100.0
        
        box_width = d["Ancho_px"]
        box_height = d["Alto_px"]
        box_area = box_width * box_height
        aspect_ratio = box_width / max(box_height, 1)
        inverse_aspect_ratio = box_height / max(box_width, 1)
        
        reject = False
        
        if cls_name == "humidity":
            if box_area > (image_area * 0.05) and (aspect_ratio > 3.0 or inverse_aspect_ratio > 3.0):
                reject = True
            elif box_area < (image_area * 0.002):
                reject = True
            elif aspect_ratio > 4.5 or inverse_aspect_ratio > 4.5:
                umbral_min_crack = umbrales_clase.get("crack", 0.25)
                if conf > umbral_min_crack:
                    d["Clase"] = "crack"
                    cls_name = "crack"
                else:
                    reject = True
        
        elif cls_name == "spalling":
            umbral_min_spalling = umbrales_clase.get("spalling", 0.50)
            if conf < umbral_min_spalling:
                reject = True
        
        elif cls_name == "crack":
            umbral_min_crack = umbrales_clase.get("crack", 0.25)
            if conf < umbral_min_crack:
                reject = True
        
        if not reject:
            d["Confianza"] = round(float(conf), 4)
            filtered_detections.append(d)
    
    return filtered_detections

def dibujar_cajas(img_cv2, detections):
    """Dibuja bounding boxes con etiquetas y colores de gravedad"""
    color_map = {"crack": (0, 0, 220), "humidity": (240, 100, 30), "spalling": (0, 140, 235)}
    
    for d in detections:
        color = color_map.get(d["Clase"], (128, 128, 128))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 255, 255), 12)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 8)
        
        conf_porcentaje = d["Confianza"] * 100
        gravity_label = d.get("gravedad_label", "")
        label = f"{d['Clase']} {conf_porcentaje:.1f}% {gravity_label}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)[0]
        label_rect_origin = (x1, max(y1 - label_size[1] - 10, label_size[1] + 10))
        label_text_origin = (x1, max(y1, label_size[1] + 10))
        
        cv2.rectangle(img_cv2, 
                      (label_rect_origin[0], label_rect_origin[1] - label_size[1]),
                      (label_rect_origin[0] + label_size[0], label_rect_origin[1]), 
                      (255, 255, 255), -1)
        cv2.putText(img_cv2, label, label_text_origin, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    
    return img_cv2

def split_image_into_tiles(image_np, tile_size, overlap):
    """Divide imagen en tiles"""
    h, w = image_np.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            if y + tile_size > h:
                y = h - tile_size
            if x + tile_size > w:
                x = w - tile_size
            
            tile = image_np[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, x, y))
            
            if x + tile_size == w:
                break
        if y + tile_size == h:
            break
    
    return tiles

def nms_boxes(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression por clase"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes).astype(np.float32)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def procesar_imagen_completa(model, image_path, class_thresholds, iou_threshold,
                              cm_per_pixel, tile_size=640, overlap=0.2, use_tiling=True,
                              progress_callback=None):
    """Motor completo con tiling + filtros geométricos"""
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return {"success": False, "error": "No se pudo cargar la imagen"}
    
    h, w = img_cv2.shape[:2]
    all_detections_raw = []
    inference_log = []
    
    if use_tiling:
        tiles = split_image_into_tiles(img_cv2, tile_size, overlap)
        total_tiles = len(tiles)
        
        for idx, (tile, x_off, y_off) in enumerate(tiles):
            if progress_callback:
                progress_callback(idx / total_tiles)
            
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            
            try:
                t0 = time.time()
                results = model.predict(
                    source=tile_rgb, imgsz=tile_size, conf=0.05, iou=iou_threshold,
                    verbose=False, show=False, augment=True,
                    agnostic_nms=False, retina_masks=True,
                    half=torch.cuda.is_available()
                )
                elapsed = (time.time() - t0) * 1000
                n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
                inference_log.append({
                    "tile": (x_off, y_off),
                    "detections": n_dets,
                    "time_ms": round(elapsed, 1)
                })
                
                if results[0].boxes is None:
                    continue
                
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0])
                    
                    if conf < class_thresholds.get(cls_name, 0.05):
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 += x_off
                    y1 += y_off
                    x2 += x_off
                    y2 += y_off
                    
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    all_detections_raw.append({
                        "Clase": cls_name,
                        "Confianza": conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "Ancho_px": x2 - x1,
                        "Alto_px": y2 - y1,
                    })
            except Exception as e:
                inference_log.append({
                    "tile": (x_off, y_off),
                    "error": str(e),
                    "status": "failed"
                })
        
        if progress_callback:
            progress_callback(1.0)
        
        if all_detections_raw:
            boxes_by_class = {}
            scores_by_class = {}
            
            for d in all_detections_raw:
                cls = d["Clase"]
                boxes_by_class.setdefault(cls, []).append([d["x1"], d["y1"], d["x2"], d["y2"]])
                scores_by_class.setdefault(cls, []).append(d["Confianza"])
            
            final_detections = []
            for cls, boxes in boxes_by_class.items():
                scores = scores_by_class[cls]
                keep = nms_boxes(boxes, scores, iou_threshold)
                
                for idx in keep:
                    d = {
                        "Clase": cls,
                        "Confianza": scores[idx],
                        "x1": boxes[idx][0], "y1": boxes[idx][1],
                        "x2": boxes[idx][2], "y2": boxes[idx][3],
                        "Ancho_px": boxes[idx][2] - boxes[idx][0],
                        "Alto_px": boxes[idx][3] - boxes[idx][1],
                    }
                    final_detections.append(d)
        else:
            final_detections = []
    else:
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        imgsz_options = [1536, 1280, 960, 640] if max(w, h) > 800 else [1280, 640]
        
        results = None
        for imgsz in imgsz_options:
            try:
                t0 = time.time()
                results = model.predict(
                    source=img_rgb, imgsz=imgsz, conf=0.05, iou=iou_threshold,
                    verbose=False, show=False, augment=True,
                    agnostic_nms=False, retina_masks=True,
                    half=torch.cuda.is_available()
                )
                n = len(results[0].boxes) if results[0].boxes is not None else 0
                inference_log.append({
                    "imgsz": imgsz,
                    "time_ms": round((time.time() - t0) * 1000, 1),
                    "detections": n,
                    "status": "success"
                })
                if n > 0:
                    break
            except Exception as e:
                inference_log.append({
                    "imgsz": imgsz,
                    "error": str(e),
                    "status": "failed"
                })
        
        final_detections = []
        if results is not None and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                
                if conf < class_thresholds.get(cls_name, 0.25):
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                final_detections.append({
                    "Clase": cls_name,
                    "Confianza": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "Ancho_px": x2 - x1,
                    "Alto_px": y2 - y1,
                })
    
    detections_filtradas = filtrar_falsos_positivos(final_detections, img_cv2.shape, class_thresholds)
    
    for d in detections_filtradas:
        area_cm2 = (d["Ancho_px"] * d["Alto_px"]) * (cm_per_pixel ** 2)
        d["Area_cm2"] = round(area_cm2, 2)
        d["Area_m2"] = round(area_cm2 / 10000, 5)
    
    # Enriquecer con gravedad
    detections_filtradas = enrich_detections_with_gravity(detections_filtradas)
    
    class_counts = {"crack": 0, "humidity": 0, "spalling": 0}
    total_area_cm2 = 0
    confidences = []
    
    for d in detections_filtradas:
        class_counts[d["Clase"]] += 1
        total_area_cm2 += d["Area_cm2"]
        confidences.append(d["Confianza"])
    
    img_con_cajas = dibujar_cajas(img_cv2.copy(), detections_filtradas)
    img_rgb = cv2.cvtColor(img_con_cajas, cv2.COLOR_BGR2RGB)
    
    preprocessing = f"Tiling (tile={tile_size}px, overlap={int(overlap*100)}%) + filtro geometrico" if use_tiling else "Nativa con filtro geometrico"
    
    return {
        "success": True,
        "annotated_image": img_rgb,
        "detections": detections_filtradas,
        "class_counts": class_counts,
        "total_area_cm2": round(total_area_cm2, 2),
        "total_area_m2": round(total_area_cm2 / 10000, 4),
        "n_detections": len(detections_filtradas),
        "avg_confidence": round(np.mean(confidences) * 100, 1) if confidences else 0,
        "preprocessing": preprocessing,
        "inference_log": inference_log,
    }

# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

class HeritageDetectorDesktop(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Heritage Damage Detector v6.5 - Desktop Edition | UCSM")
        self.geometry("1500x900")
        self.minsize(1200, 700)
        
        self.model = None
        self.db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_URL and SUPABASE_KEY:
            ok, msg = self.db.ping()
            if not ok:
                messagebox.showwarning("Supabase", f"No se pudo conectar a Supabase:\n{msg}\n\nLa app funcionará pero no guardará datos.")
        else:
            messagebox.showwarning("Supabase", "Sin credenciales de Supabase.\nConfigura SUPABASE_URL y SUPABASE_KEY en .env\n\nLa app funcionará pero no guardará datos.")
        
        self.current_image_path = None
        self.last_result = None
        self.current_module = "Analisis"
        
        # Variables para procesamiento por lotes
        self.processing_queue = []
        self.current_queue_index = -1
        self.all_results = []
        self.current_view_index = 0
        
        self.class_thresholds = {"crack": 0.20, "humidity": 0.50, "spalling": 0.60}
        self.iou_threshold = 0.45
        self.cm_per_pixel = 0.13
        self.use_tiling = True
        self.tile_size = 640
        self.overlap = 0.20
        
        self.load_model()
        self.create_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.alert_system = AlertSystem(thresholds={
            "max_damage_area_m2": 0.5,
            "min_new_cracks": 10,
            "growth_percentage_threshold": 20,
            "min_avg_confidence": 70.0
        })
    
    def load_model(self):
        try:
            if not Path(MODEL_PATH).exists():
                messagebox.showerror("Error", f"No se encontró {MODEL_PATH}\n\nColoca best.pt en la misma carpeta.")
                return
            
            self.model = YOLO(MODEL_PATH)
            _ = self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, imgsz=640)
        except Exception:
            messagebox.showerror("Error al cargar modelo", "No se pudo cargar el modelo best.pt. Verifica que el archivo sea válido.")
    
    def create_ui(self):
        self.configure(fg_color=COLORS["bg_primary"])
        
        self.sidebar = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"], width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=15, pady=(20, 10))
        
        ctk.CTkLabel(brand_frame, text="HERITAGE", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(brand_frame, text="DETECTOR", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["accent_gold"]).pack()
        ctk.CTkLabel(brand_frame, text="v6.5 · Desktop · UCSM", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(5, 0))
        
        separator = ctk.CTkFrame(self.sidebar, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator.pack(fill="x", padx=20, pady=15)
        
        self.nav_buttons = {}
        modules = [
            ("Analisis", "Análisis en Campo"),
            ("Dashboard", "Cuadro de Mandos"),
            ("Historial", "Historial"),
            ("Especificacion", "Especificación Técnica"),
        ]
        
        for key, label in modules:
            btn = ctk.CTkButton(
                self.sidebar, text=label,
                command=lambda k=key: self.switch_module(k),
                font=(FONT_FAMILY, 12, "bold"),
                fg_color="transparent",
                text_color=COLORS["text_primary"],
                hover_color=COLORS["bg_surface"],
                anchor="w", height=40
            )
            btn.pack(fill="x", padx=15, pady=3)
            self.nav_buttons[key] = btn
        
        separator2 = ctk.CTkFrame(self.sidebar, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator2.pack(fill="x", padx=20, pady=15)
        
        ok, msg = self.db.ping()
        status_text = "Supabase conectado" if ok else "Supabase offline"
        status_color = COLORS["success"] if ok else COLORS["danger"]
        status_bg = COLORS["success_light"] if ok else COLORS["bg_surface"]
        db_status = ctk.CTkFrame(self.sidebar, fg_color=status_bg, corner_radius=8, border_width=1, border_color=COLORS["border"])
        db_status.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(db_status, text=status_text, font=(FONT_FAMILY, 10, "bold"), text_color=status_color).pack(pady=8)
        
        ctk.CTkLabel(self.sidebar, text="Santuario de San Agustin\nArequipa, Peru", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"], justify="center").pack(side="bottom", pady=15)
        
        self.content = ctk.CTkFrame(self, fg_color=COLORS["bg_primary"])
        self.content.pack(side="right", fill="both", expand=True)
        
        self.show_module_analisis()
    
    def switch_module(self, module_name):
        self.current_module = module_name
        
        for widget in self.content.winfo_children():
            widget.destroy()
        
        for key, btn in self.nav_buttons.items():
            if key == module_name:
                btn.configure(fg_color=COLORS["bg_surface"], text_color=COLORS["text_primary"], border_width=3, border_color=COLORS["primary"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text_primary"], border_width=0)
        
        if module_name == "Analisis":
            self.show_module_analisis()
        elif module_name == "Dashboard":
            self.show_module_dashboard()
        elif module_name == "Historial":
            self.show_module_historial()
        elif module_name == "Especificacion":
            self.show_module_especificacion()
    
    def show_module_analisis(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="INSPECCIÓN ESTRUCTURAL", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Análisis de Sillar Volcánico", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Filtrado geométrico anti-porosidad + anti-sombras + micro-inferencia por tiling + HOVER INTERACTIVO", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        main = ctk.CTkFrame(self.content, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)
        
        left = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], width=380, corner_radius=12, border_width=1, border_color=COLORS["border"])
        left.pack(side="left", fill="both", padx=(0, 10))
        left.pack_propagate(False)
        
        right = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        right.pack(side="right", fill="both", expand=True)
        
        self.create_control_panel(left)
        self.create_visualization_panel(right)
    
    def create_control_panel(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg_card"])
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(scroll, text="CARGA DE IMAGEN", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        self.btn_load = ctk.CTkButton(
            scroll, text="Cargar Imagen(es) del Monumento",
            command=self.load_image,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            height=40
        )
        self.btn_load.pack(pady=5, fill="x")
        
        self.btn_cancel = ctk.CTkButton(
            scroll, text="CANCELAR PROCESAMIENTO",
            command=self.cancel_processing,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["danger"],
            text_color=COLORS["danger"], hover_color=COLORS["bg_surface"],
            height=35
        )
        self.btn_cancel.pack(pady=5, fill="x")
        
        self.lbl_file = ctk.CTkLabel(scroll, text="Ningún archivo cargado", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"])
        self.lbl_file.pack(pady=(0, 10))
        
        separator = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator.pack(fill="x", pady=8)
        
        ctk.CTkLabel(scroll, text="CALIBRACIÓN PARA SILLAR", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        ctk.CTkLabel(scroll, text="Grietas (Crack):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_crack = ctk.CTkSlider(scroll, from_=0.05, to=0.90, number_of_steps=17, command=lambda v: self.update_label(self.lbl_crack, v))
        self.slider_crack.set(0.20)
        self.slider_crack.pack(fill="x", pady=2)
        self.lbl_crack = ctk.CTkLabel(scroll, text="20%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["crack"])
        self.lbl_crack.pack()
        
        ctk.CTkLabel(scroll, text="Humedad (Humidity):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_humidity = ctk.CTkSlider(scroll, from_=0.05, to=0.90, number_of_steps=17, command=lambda v: self.update_label(self.lbl_humidity, v))
        self.slider_humidity.set(0.50)
        self.slider_humidity.pack(fill="x", pady=2)
        self.lbl_humidity = ctk.CTkLabel(scroll, text="50%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["humidity"])
        self.lbl_humidity.pack()
        
        ctk.CTkLabel(scroll, text="Desprendimiento (Spalling):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_spalling = ctk.CTkSlider(scroll, from_=0.05, to=0.90, number_of_steps=17, command=lambda v: self.update_label(self.lbl_spalling, v))
        self.slider_spalling.set(0.60)
        self.slider_spalling.pack(fill="x", pady=2)
        self.lbl_spalling = ctk.CTkLabel(scroll, text="60%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["spalling"])
        self.lbl_spalling.pack()
        
        ctk.CTkLabel(scroll, text="IoU Threshold:", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_iou = ctk.CTkSlider(scroll, from_=0.1, to=0.9, number_of_steps=16, command=lambda v: self.update_label(self.lbl_iou, v))
        self.slider_iou.set(0.45)
        self.slider_iou.pack(fill="x", pady=2)
        self.lbl_iou = ctk.CTkLabel(scroll, text="45%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_primary"])
        self.lbl_iou.pack()
        
        ctk.CTkLabel(scroll, text="GSD (cm/pixel):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.entry_gsd = ctk.CTkEntry(scroll, width=100, font=(FONT_FAMILY, 11), fg_color=COLORS["bg_surface"], border_color=COLORS["border"], text_color=COLORS["text_primary"])
        self.entry_gsd.insert(0, "0.13")
        self.entry_gsd.pack(pady=2)
        
        ctk.CTkButton(
            scroll, text="↺ Restaurar calibración recomendada",
            command=self.restore_recommended_calibration,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], height=28
        ).pack(fill="x", pady=(8, 2))
        
        separator2 = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator2.pack(fill="x", pady=8)
        
        ctk.CTkLabel(scroll, text="MICRO-INFERENCIA", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        self.chk_tiling = ctk.CTkCheckBox(scroll, text="Activar Tiling (detalles pequeños)", command=self.toggle_tiling, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"])
        self.chk_tiling.select()
        self.chk_tiling.pack(anchor="w", pady=5)
        
        ctk.CTkLabel(scroll, text="Tamaño del tile (px):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_tile = ctk.CTkSlider(scroll, from_=320, to=1280, number_of_steps=15, command=lambda v: self.update_label(self.lbl_tile, v, is_int=True))
        self.slider_tile.set(640)
        self.slider_tile.pack(fill="x", pady=2)
        self.lbl_tile = ctk.CTkLabel(scroll, text="640 px", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_primary"])
        self.lbl_tile.pack()
        
        ctk.CTkLabel(scroll, text="Solapamiento (%):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_overlap = ctk.CTkSlider(scroll, from_=0, to=50, number_of_steps=10, command=lambda v: self.update_label(self.lbl_overlap, v, is_int=True))
        self.slider_overlap.set(20)
        self.slider_overlap.pack(fill="x", pady=2)
        self.lbl_overlap = ctk.CTkLabel(scroll, text="20%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_primary"])
        self.lbl_overlap.pack()
        
        separator3 = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator3.pack(fill="x", pady=8)
        
        self.btn_process = ctk.CTkButton(
            scroll, text="INICIAR PROCESAMIENTO",
            command=self.process_image,
            font=(FONT_FAMILY, 12, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            height=45, state="disabled"
        )
        self.btn_process.pack(pady=10, fill="x")
        
        self.progress_bar = ctk.CTkProgressBar(scroll, height=8, fg_color=COLORS["bg_surface"], progress_color=COLORS["accent_gold"])
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)
        
        self.btn_save = ctk.CTkButton(
            scroll, text="GUARDAR EN BASE DE DATOS",
            command=self.save_results,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLORS["success"], hover_color=COLORS["primary_hover"],
            height=40, state="disabled"
        )
        self.btn_save.pack(pady=5, fill="x")
        
        self.nav_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        self.nav_frame.pack(fill="x", pady=10)
        
        self.btn_previous = ctk.CTkButton(
            self.nav_frame, text="◀ Anterior",
            command=self.show_previous_image,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color=COLORS["bg_surface"], hover_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            height=35, state="disabled", width=120
        )
        self.btn_previous.pack(side="left", padx=(0, 5))
        
        self.lbl_image_counter = ctk.CTkLabel(
            self.nav_frame, text="Imagen 0 de 0",
            font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_primary"]
        )
        self.lbl_image_counter.pack(side="left", padx=5)
        
        self.btn_next = ctk.CTkButton(
            self.nav_frame, text="Siguiente ▶",
            command=self.show_next_image,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color=COLORS["bg_surface"], hover_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            height=35, state="disabled", width=120
        )
        self.btn_next.pack(side="left", padx=(5, 0))
        
        self.btn_export = ctk.CTkButton(
            scroll, text="EXPORTAR CSV",
            command=self.export_csv,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["border"],
            text_color=COLORS["text_primary"], hover_color=COLORS["bg_surface"],
            height=40, state="disabled"
        )
        self.btn_export.pack(pady=5, fill="x")
        
        separator4 = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator4.pack(fill="x", pady=8)
        
        ctk.CTkLabel(scroll, text="RESULTADOS", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        self.results_frame = ctk.CTkFrame(scroll, fg_color=COLORS["bg_surface"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        self.results_frame.pack(fill="both", expand=True, pady=5)
        
        self.lbl_waiting = ctk.CTkLabel(self.results_frame, text="Esperando imagen...", font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"])
        self.lbl_waiting.pack(pady=20)
    
    def create_visualization_panel(self, parent):
        """Panel de visualización con Canvas interactivo para hover"""
        self.viz_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_surface"], corner_radius=8)
        self.viz_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.canvas = tk.Canvas(
            self.viz_frame, 
            bg=COLORS["bg_surface"], 
            highlightthickness=0
        )
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.tooltip = tk.Toplevel(self.canvas)
        self.tooltip.withdraw()
        self.tooltip.overrideredirect(True)
        self.tooltip.configure(bg="#1E293B", padx=8, pady=6)
        
        self.tooltip_label = tk.Label(
            self.tooltip, 
            text="", 
            bg="#1E293B", 
            fg="#FFFFFF", 
            font=("Segoe UI", 10, "bold"),
            justify="left"
        )
        self.tooltip_label.pack()
        
        self.canvas.create_text(
            400, 300,
            text="Carga una imagen del monumento para comenzar\n\n• Pasa el cursor sobre las detecciones para ver medidas\n• Colores indican gravedad: 🟢 Leve → 🔴 Crítica",
            fill=COLORS["text_muted"],
            font=("Segoe UI", 14),
            justify="center"
        )
        
        self.drawn_boxes = []
    
    def display_image(self, path_or_array):
        """Muestra imagen con detecciones INTERACTIVAS (hover = medidas)"""
        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array)
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = path_or_array
    
        canvas_w = self.canvas.winfo_width() or 1000
        canvas_h = self.canvas.winfo_height() or 700
        h, w = img.shape[:2]
        ratio = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * ratio), int(h * ratio)
    
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
    
        # ✅ CORRECCIÓN: Usar ImageTk.PhotoImage en lugar de CTkImage
        self.tk_image = ImageTk.PhotoImage(img_pil)
    
        self.canvas.delete("all")
        self.drawn_boxes = []
    
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        offset_x = (cw - new_w) // 2
        offset_y = (ch - new_h) // 2
    
        # ✅ Ahora sí funciona con tkinter Canvas
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)
    
        self._current_display_ratio = ratio
        self._current_display_offset = (offset_x, offset_y)
    
        if self.last_result and self.last_result.get("detections"):
            for d in self.last_result["detections"]:
                x1 = int(d["x1"] * ratio) + offset_x
                y1 = int(d["y1"] * ratio) + offset_y
                x2 = int(d["x2"] * ratio) + offset_x
                y2 = int(d["y2"] * ratio) + offset_y
            
                gravity = d.get("gravedad", "leve")
                color_hex = GRAVITY_COLORS.get(gravity, "#DC2626")
                color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color_str = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            
                rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color_str, width=3, fill=""
                )
            
                label_text = f"{d['Clase']} {d.get('gravedad_label', '')} | {d['Area_cm2']:.1f}cm²"
                label_id = self.canvas.create_text(
                x1, max(y1 - 12, 10),
                text=label_text,
                anchor="sw",
                fill=color_str,
                font=("Segoe UI", 9, "bold")
               )   
            
                self.drawn_boxes.append({
                "coords": (x1, y1, x2, y2),
                "detection": d,
                "rect_id": rect_id,
                "label_id": label_id,
                "color": color_str
                })
    
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
    def _on_canvas_motion(self, event):
        """Muestra tooltip con medidas exactas al pasar el cursor"""
        if not self.drawn_boxes:
            return
        
        hovered_box = None
        for box in self.drawn_boxes:
            x1, y1, x2, y2 = box["coords"]
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                hovered_box = box
                break
        
        if hovered_box:
            d = hovered_box["detection"]
            
            self.canvas.itemconfig(hovered_box["rect_id"], width=5)
            
            tooltip_text = (
                f"📐 {d['Clase'].upper()}\n"
                f"🎯 Gravedad: {d.get('gravedad_label', 'N/A')}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📏 Área: {d['Area_cm2']:.2f} cm²\n"
                f"      ({d['Area_m2']:.5f} m²)\n"
                f"↔️  Ancho: {d['Ancho_px']} px\n"
                f"↕️  Alto:  {d['Alto_px']} px\n"
                f"🎯 Confianza: {d['Confianza']*100:.1f}%"
            )
            
            self.tooltip_label.configure(text=tooltip_text)
            
            canvas_root_x = self.canvas.winfo_rootx()
            canvas_root_y = self.canvas.winfo_rooty()
            self.tooltip.geometry(f"+{canvas_root_x + event.x + 15}+{canvas_root_y + event.y + 15}")
            self.tooltip.deiconify()
        else:
            for box in self.drawn_boxes:
                self.canvas.itemconfig(box["rect_id"], width=3)
            self.tooltip.withdraw()
    
    def _on_canvas_leave(self, event):
        """Oculta tooltip al salir del canvas"""
        self.tooltip.withdraw()
        for box in self.drawn_boxes:
            self.canvas.itemconfig(box["rect_id"], width=3)
    
    def update_label(self, label, value, is_int=False):
        if is_int:
            label.configure(text=f"{int(value)} px" if "tile" in str(label) else f"{int(value)}%")
        else:
            label.configure(text=f"{int(value*100)}%")
    
    def toggle_tiling(self):
        self.use_tiling = self.chk_tiling.get()

    def restore_recommended_calibration(self):
        """Restaura la calibracion optima recomendada para sillar."""
        self.slider_crack.set(0.20)
        self.slider_humidity.set(0.50)
        self.slider_spalling.set(0.60)
        self.slider_iou.set(0.45)
        self.update_label(self.lbl_crack, 0.20)
        self.update_label(self.lbl_humidity, 0.50)
        self.update_label(self.lbl_spalling, 0.60)
        self.update_label(self.lbl_iou, 0.45)
        messagebox.showinfo("Calibración", "Calibración restablecida a los valores óptimos recomendados para sillar (Crack 20%, Humedad 50%, Desprendimiento 60%).")
    
    def load_image(self):
        file_paths = filedialog.askopenfilenames(
            title="Seleccionar Imágenes del Monumento",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        # Limite de tamano por imagen (20 MB)
        MAX_IMAGE_MB = 20
        oversized = [os.path.basename(fp) for fp in file_paths if os.path.getsize(fp) > MAX_IMAGE_MB * 1024 * 1024]
        if oversized:
            messagebox.showerror(
                "Imagenes Demasiado Grandes",
                f"El peso maximo por imagen es de {MAX_IMAGE_MB} MB.\n\n"
                f"Archivo(s) excedido(s):\n{chr(10).join(oversized)}"
            )
            return
        
        filenames = [os.path.basename(fp) for fp in file_paths]
        
        try:
            validation = self.db.check_duplicate_files(filenames)
            duplicates = validation.get("duplicates", [])
            new_files = validation.get("new_files", [])
            
            if duplicates:
                dup_list = "\n".join(duplicates)
                continuar = messagebox.askyesno(
                    "Archivos Duplicados Detectados",
                    f"Los siguientes archivos YA EXISTEN en la base de datos:\n\n{dup_list}\n\n"
                    f"Se ignorarán y se procesarán los {len(new_files)} archivos nuevos.\n\n"
                    "¿Deseas continuar?"
                )
                
                if not continuar:
                    return
            
            if not new_files:
                messagebox.showinfo(
                    "Sin Archivos Nuevos",
                    "Todos los archivos seleccionados ya existen en la base de datos."
                )
                return
            
            self.processing_queue = [
                fp for fp in file_paths
                if os.path.basename(fp) in new_files
            ]
            
            self.current_queue_index = -1
            self.all_results = []
            self.current_view_index = 0
            
            self.lbl_file.configure(
                text=f"{len(self.processing_queue)} archivos listos para procesar",
                text_color=COLORS["success"]
            )
            
            self.btn_process.configure(state="normal", text="INICIAR PROCESAMIENTO")
            self.btn_save.configure(state="disabled", text="GUARDAR EN BASE DE DATOS")
            self.btn_export.configure(state="disabled")
            self.btn_previous.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            self.lbl_image_counter.configure(text="Imagen 0 de 0")
            
            messagebox.showinfo(
                "Cola Preparada",
                f"✅ {len(self.processing_queue)} archivos nuevos listos.\n\n"
                f"Haz clic en 'INICIAR PROCESAMIENTO' para comenzar.\n"
                f"El botón GUARDAR se habilitará al terminar todo el lote."
            )
        
        except Exception:
            messagebox.showerror("Error de Conexión", "No se pudo verificar duplicados con Supabase.")
    
    def _start_batch_processing(self):
        if not self.processing_queue:
            return
        
        self.current_queue_index += 1
        
        if self.current_queue_index >= len(self.processing_queue):
            self._on_batch_finished()
            return
        
        next_path = self.processing_queue[self.current_queue_index]
        filename = os.path.basename(next_path)
        total = len(self.processing_queue)
        actual = self.current_queue_index + 1
        
        self.lbl_file.configure(
            text=f"Procesando {filename} ({actual}/{total})",
            text_color=COLORS["accent_gold"]
        )
        self.btn_process.configure(state="disabled", text=f"PROCESANDO {actual}/{total}...")
        self.btn_load.configure(state="disabled")
        self.btn_save.configure(state="disabled", text="GUARDAR EN BASE DE DATOS")
        
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.lbl_waiting = ctk.CTkLabel(
            self.results_frame,
            text=f"Procesando imagen {actual} de {total}...\n{filename}",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"]
        )
        self.lbl_waiting.pack(pady=20)
        
        self.class_thresholds = {
            "crack": self.slider_crack.get(),
            "humidity": self.slider_humidity.get(),
            "spalling": self.slider_spalling.get()
        }
        self.iou_threshold = self.slider_iou.get()
        
        try:
            self.cm_per_pixel = float(self.entry_gsd.get())
        except:
            self.cm_per_pixel = 0.13
        
        self.tile_size = int(self.slider_tile.get())
        self.overlap = self.slider_overlap.get() / 100.0
        
        def worker():
            result = procesar_imagen_completa(
                self.model, next_path,
                self.class_thresholds, self.iou_threshold,
                self.cm_per_pixel, self.tile_size, self.overlap,
                use_tiling=self.use_tiling,
                progress_callback=lambda p: self.after(0, lambda: self.progress_bar.set(p))
            )
            self.after(0, lambda: self._on_single_process_done(result, next_path))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _on_single_process_done(self, result, image_path):
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        
        if not result["success"]:
            messagebox.showerror(
                "Error de Procesamiento",
                f"No se pudo procesar {os.path.basename(image_path)}:\n{result.get('error', 'Error desconocido')}\n\n"
                "Se continuará con la siguiente imagen."
            )
            self.after(500, self._start_batch_processing)
            return
        
        # Enriquecer con reporte de priorización
        if result.get("detections"):
            result["priority_report"] = generate_priority_report(
                result["detections"], 
                os.path.basename(image_path)
            )
        
        last_history = None
        filename = os.path.basename(image_path)
        hist = self.db.select("inspection_results", "*", filters={"filename": f"eq.{filename}"}, order="created_at.desc", limit=1)
        if hist["success"] and hist["data"]:
            last_history = hist["data"][0]
        
        alerts = self.alert_system.evaluate_current_result(result, last_history)
        if alerts:
            self.alert_system.log_alerts(alerts)
        
        result["filename"] = os.path.basename(image_path)
        result["timestamp"] = datetime.now().isoformat()
        result["image_path"] = image_path
        
        self.all_results.append(result)
        self.last_result = result
        
        self.display_image(result["annotated_image"])
        self.show_results_visual(result)
        
        self.after(500, self._start_batch_processing)
    
    def _on_batch_finished(self):
        """Se llama cuando termina toda la cola - NO guarda automáticamente"""
        total = len(self.processing_queue)
        processed = len(self.all_results)
        
        self.processing_queue.clear()
        self.current_queue_index = -1
        
        self.btn_process.configure(state="normal", text="INICIAR PROCESAMIENTO")
        self.btn_load.configure(state="normal")
        
        if processed > 0:
            self.btn_save.configure(
                state="normal",
                text=f"GUARDAR {processed} RESULTADOS"
            )
            self.btn_export.configure(state="normal")
            
            self.current_view_index = 0
            
            if processed > 1:
                self.btn_previous.configure(state="normal")
                self.btn_next.configure(state="normal")
            else:
                self.btn_previous.configure(state="disabled")
                self.btn_next.configure(state="disabled")
            
            self.update_image_counter()
            
            self.last_result = self.all_results[0]
            self.display_image(self.last_result["annotated_image"])
            self.show_results_visual(self.last_result)
            
            messagebox.showinfo(
                "Procesamiento Completado",
                f"✅ Se procesaron {processed} imágenes exitosamente.\n\n"
                f"📊 Usa los botones ◀ Anterior y Siguiente ▶ para navegar.\n\n"
                f"💡 Pasa el cursor sobre las detecciones para ver medidas exactas.\n\n"
                f"Haz clic en 'GUARDAR {processed} RESULTADOS' para subirlos a Supabase."
            )
        else:
            self.btn_save.configure(state="disabled")
            self.btn_export.configure(state="disabled")
            messagebox.showwarning(
                "Sin Resultados",
                "No se pudo procesar ninguna imagen de la cola."
            )
    
    def process_image(self):
        if not self.processing_queue:
            messagebox.showwarning("Sin cola", "No hay imágenes en la cola.\nCarga imágenes primero.")
            return
        
        if self.current_queue_index >= 0:
            messagebox.showwarning("En progreso", "Ya hay un procesamiento en curso.")
            return
        
        self.progress_bar.set(0)
        self._start_batch_processing()
    
    def show_previous_image(self):
        if not self.all_results:
            return
        
        if self.current_view_index > 0:
            self.current_view_index -= 1
            self.update_display_for_current_image()
    
    def show_next_image(self):
        if not self.all_results:
            return
        
        if self.current_view_index < len(self.all_results) - 1:
            self.current_view_index += 1
            self.update_display_for_current_image()
    
    def update_display_for_current_image(self):
        if not self.all_results or self.current_view_index >= len(self.all_results):
            return
        
        result = self.all_results[self.current_view_index]
        self.last_result = result
        
        self.display_image(result["annotated_image"])
        self.show_results_visual(result)
        self.update_image_counter()
        
        if self.current_view_index == 0:
            self.btn_previous.configure(state="disabled")
        else:
            self.btn_previous.configure(state="normal")
        
        if self.current_view_index == len(self.all_results) - 1:
            self.btn_next.configure(state="disabled")
        else:
            self.btn_next.configure(state="normal")
    
    def update_image_counter(self):
        total = len(self.all_results)
        current = self.current_view_index + 1
        self.lbl_image_counter.configure(text=f"Imagen {current} de {total}")
    
    def cancel_processing(self):
        """Cancela el procesamiento de la cola restante"""
        if not self.processing_queue or self.current_queue_index == -1:
            messagebox.showinfo("Sin cola", "No hay procesamiento en curso.")
            return
        
        pendientes = len(self.processing_queue) - self.current_queue_index - 1
        
        if pendientes <= 0:
            messagebox.showinfo("Sin pendientes", "No hay imágenes pendientes en la cola.")
            return
        
        confirmar = messagebox.askyesno(
            "Cancelar Procesamiento",
            f"Se cancelarán {pendientes} imágenes pendientes.\n"
            f"Las {self.current_queue_index + 1} ya procesadas se mantendrán.\n\n"
            "¿Deseas continuar?"
        )
        
        if confirmar:
            self.processing_queue = self.processing_queue[:self.current_queue_index + 1]
            self._on_batch_finished()
    
    def show_results_visual(self, result):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.results_frame,
            text=f"Archivo: {result['filename']}",
            font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        metrics_grid = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=10, pady=5)
        
        metric_card = ctk.CTkFrame(metrics_grid, fg_color=COLORS["bg_card"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        metric_card.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        metric_card.columnconfigure(0, weight=1)
        ctk.CTkLabel(metric_card, text=str(result['n_detections']), font=(FONT_FAMILY, 24, "bold"), text_color=COLORS["primary"]).pack(pady=(10, 2))
        ctk.CTkLabel(metric_card, text="Total Detecciones", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(0, 10))
        
        metric_card2 = ctk.CTkFrame(metrics_grid, fg_color=COLORS["bg_card"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        metric_card2.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        metric_card2.columnconfigure(0, weight=1)
        ctk.CTkLabel(metric_card2, text=f"{result['total_area_cm2']:.1f} cm²", font=(FONT_FAMILY, 24, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 2))
        ctk.CTkLabel(metric_card2, text="Área Afectada", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(0, 10))
        
        metric_card3 = ctk.CTkFrame(metrics_grid, fg_color=COLORS["bg_card"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        metric_card3.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        metric_card3.columnconfigure(0, weight=1)
        ctk.CTkLabel(metric_card3, text=f"{result['avg_confidence']:.1f}%", font=(FONT_FAMILY, 24, "bold"), text_color=COLORS["success"]).pack(pady=(10, 2))
        ctk.CTkLabel(metric_card3, text="Confianza Promedio", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(0, 10))
        
        separator = ctk.CTkFrame(self.results_frame, height=2, fg_color=COLORS["border"], corner_radius=1)
        separator.pack(fill="x", padx=10, pady=10)
        
        # Mostrar reporte de priorización si existe
        if result.get("priority_report"):
            pr = result["priority_report"]
            
            priority_frame = ctk.CTkFrame(
                self.results_frame, 
                fg_color=COLORS["bg_card"], 
                corner_radius=8, 
                border_width=2,
                border_color=GRAVITY_COLORS.get(
                    "critica" if pr["urgency"] == "CRÍTICA" else 
                    "severa" if pr["urgency"] == "ALTA" else "moderada",
                    "#10B981"
                )
            )
            priority_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                priority_frame,
                text=f"🏛️ PRIORIZACIÓN DE INTERVENCIÓN",
                font=(FONT_FAMILY, 11, "bold"),
                text_color=COLORS["accent_gold"]
            ).pack(pady=(10, 5))
            
            ctk.CTkLabel(
                priority_frame,
                text=pr["recommendation"],
                font=(FONT_FAMILY, 12, "bold"),
                text_color=COLORS["text_primary"]
            ).pack(pady=5)
            
            info_text = (
                f"Score prioridad: {pr['total_priority_score']} | "
                f"Urgencia: {pr['urgency']} | "
                f"Críticas: {pr['criticas_count']} | Severas: {pr['severas_count']}"
            )
            ctk.CTkLabel(
                priority_frame,
                text=info_text,
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_secondary"]
            ).pack(pady=(0, 10))
        
        ctk.CTkLabel(
            self.results_frame,
            text="Desglose por tipo:",
            font=(FONT_FAMILY, 9, "bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=10, pady=(0, 5))
        
        crack_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color=COLORS["crack"])
        crack_badge.pack(fill="x", padx=10, pady=2)
        crack_inner = ctk.CTkFrame(crack_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
        crack_inner.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(crack_inner, text=f"🔴 Grietas: {result['class_counts']['crack']}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["crack"]).pack(pady=5)
        
        humidity_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color=COLORS["humidity"])
        humidity_badge.pack(fill="x", padx=10, pady=2)
        humidity_inner = ctk.CTkFrame(humidity_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
        humidity_inner.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(humidity_inner, text=f"🔵 Humedad: {result['class_counts']['humidity']}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["humidity"]).pack(pady=5)
        
        spalling_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color=COLORS["spalling"])
        spalling_badge.pack(fill="x", padx=10, pady=2)
        spalling_inner = ctk.CTkFrame(spalling_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
        spalling_inner.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(spalling_inner, text=f"🟠 Desprendimiento: {result['class_counts']['spalling']}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["spalling"]).pack(pady=5)
    
    def save_results(self):
        """Guarda TODOS los resultados acumulados en Supabase"""
        if not self.all_results:
            messagebox.showwarning("Sin datos", "No hay resultados para guardar.")
            return
        
        confirmar = messagebox.askyesno(
            "Confirmar Guardado",
            f"Se guardarán {len(self.all_results)} inspecciones en Supabase.\n\n"
            "¿Deseas continuar?"
        )
        if not confirmar:
            return
        
        success_count = 0
        error_count = 0
        errores = []
        
        for result in self.all_results:
            success, info = self.db.save_inspection(result, result["filename"])
            if success:
                success_count += 1
                self.last_result = result
                self.save_local_files()
            else:
                error_count += 1
                errores.append(f"{result['filename']}: {info}")
        
        if success_count > 0:
            messagebox.showinfo(
                "Guardado Exitoso",
                f"✅ {success_count} inspecciones guardadas en Supabase.\n"
                f"❌ {error_count} errores." + 
                (f"\n\nDetalles de errores:\n" + "\n".join(errores[:5]) if errores else "")
            )
            self.btn_save.configure(state="disabled", text="YA GUARDADO")
            self.btn_export.configure(state="disabled")
            self.all_results.clear()
        else:
            messagebox.showerror(
                "Error al guardar",
                "No se pudo guardar ninguna inspección.\n\n" + "\n".join(errores)
            )
    
    def save_local_files(self):
        filename_base = os.path.splitext(self.last_result["filename"])[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("resultados")
        output_dir.mkdir(exist_ok=True)
        
        try:
            img_path = output_dir / f"resultado_{filename_base}_{timestamp}.jpg"
            img_bgr = cv2.cvtColor(self.last_result["annotated_image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr)
            
            json_path = output_dir / f"resultado_{filename_base}_{timestamp}.json"
            json_data = {
                "filename": self.last_result["filename"],
                "timestamp": self.last_result["timestamp"],
                "parameters": {
                    "crack_threshold": self.class_thresholds["crack"],
                    "humidity_threshold": self.class_thresholds["humidity"],
                    "spalling_threshold": self.class_thresholds["spalling"],
                    "iou_threshold": self.iou_threshold,
                    "gsd_cm_per_pixel": self.cm_per_pixel,
                    "use_tiling": self.use_tiling,
                    "tile_size": self.tile_size,
                    "overlap": self.overlap
                },
                "summary": {
                    "total_detections": self.last_result["n_detections"],
                    "crack_count": self.last_result["class_counts"]["crack"],
                    "humidity_count": self.last_result["class_counts"]["humidity"],
                    "spalling_count": self.last_result["class_counts"]["spalling"],
                    "total_area_cm2": self.last_result["total_area_cm2"],
                    "total_area_m2": self.last_result["total_area_m2"],
                    "avg_confidence": self.last_result["avg_confidence"]
                },
                "detections": [
                    {
                        "class": d["Clase"],
                        "confidence": round(d["Confianza"] * 100, 1),
                        "x1": d["x1"], "y1": d["y1"],
                        "x2": d["x2"], "y2": d["y2"],
                        "width_px": d["Ancho_px"],
                        "height_px": d["Alto_px"],
                        "area_cm2": d["Area_cm2"],
                        "area_m2": d["Area_m2"],
                        "gravedad": d.get("gravedad", "leve"),
                        "gravedad_score": d.get("gravedad_score", 1)
                    }
                    for d in self.last_result["detections"]
                ]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error guardando archivos locales: {e}")
    
    def export_csv(self):
        if not self.last_result:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exportar Reporte CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"reporte_sillar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        
        if not file_path:
            return
        
        try:
            data = [{
                "Archivo": self.last_result["filename"],
                "Cracks": self.last_result["class_counts"].get("crack", 0),
                "Humedad": self.last_result["class_counts"].get("humidity", 0),
                "Spalling": self.last_result["class_counts"].get("spalling", 0),
                "Total_Detecciones": self.last_result["n_detections"],
                "Area_cm2": self.last_result["total_area_cm2"],
                "Area_m2": self.last_result["total_area_m2"],
                "Confianza_Promedio": self.last_result.get("avg_confidence", 0),
            }]
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
            
            messagebox.showinfo("Exportación Exitosa", f"CSV guardado en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_module_dashboard(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="DASHBOARD EJECUTIVO", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Cuadro de Mandos Consolidado", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Métricas clave, distribución por tipología y superficie afectada", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        content = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)
        
        stats = self.db.get_stats()
        
        if stats["total_inspections"] == 0:
            ctk.CTkLabel(content, text="No hay datos procesados.\nEjecute primero 'Análisis en Campo' y guarde resultados.", font=(FONT_FAMILY, 14), text_color=COLORS["text_muted"]).pack(pady=50)
            return
        
        metrics_frame = ctk.CTkFrame(content, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 20))
        
        for label, value, color in [
            ("Total Inspecciones", str(stats["total_inspections"]), COLORS["primary"]),
            ("Total Daños", str(stats["total_detections"]), COLORS["crack"]),
            ("Superficie Afectada", f"{stats['total_area_m2']:.4f} m²", COLORS["accent_gold"]),
            ("Confianza Promedio", f"{stats['avg_confidence']:.1f}%", COLORS["success"]),
        ]:
            card = ctk.CTkFrame(metrics_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            card.pack(side="left", fill="both", expand=True, padx=5)
            
            bar = ctk.CTkFrame(card, height=4, fg_color=color, corner_radius=2)
            bar.pack(fill="x", padx=15, pady=(15, 5))
            
            ctk.CTkLabel(card, text=label.upper(), font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_muted"]).pack(pady=(5, 5))
            ctk.CTkLabel(card, text=value, font=(FONT_FAMILY, 28, "bold"), text_color=color).pack(pady=(0, 15))
        
        charts_frame = ctk.CTkFrame(content, fg_color="transparent")
        charts_frame.pack(fill="both", expand=True, pady=10)
        
        chart1_frame = ctk.CTkFrame(charts_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        chart1_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(chart1_frame, text="Distribución de Daños por Tipo", font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"]).pack(pady=10)
        
        fig1 = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS["bg_card"])
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor(COLORS["bg_card"])
        
        types = ["Grietas", "Humedad", "Desprendimiento"]
        counts = [stats["total_cracks"], stats["total_humidity"], stats["total_spalling"]]
        colors_bar = [COLORS["crack"], COLORS["humidity"], COLORS["spalling"]]
        
        bars = ax1.bar(types, counts, color=colors_bar, edgecolor=COLORS["bg_surface"], linewidth=2)
        ax1.set_ylabel("Cantidad", fontweight='bold', color=COLORS["text_primary"])
        ax1.set_title("")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(colors=COLORS["text_muted"])
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', fontweight='bold', fontsize=11, color=COLORS["text_primary"])
        
        canvas1 = FigureCanvasTkAgg(fig1, master=chart1_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        chart2_frame = ctk.CTkFrame(charts_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        chart2_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(chart2_frame, text="Distribución de Área Afectada (m²)", font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"]).pack(pady=10)
        
        historical = self.db.load_historical_data(1000)
        inspection_ids = [r["id"] for r in historical]
        details_by_insp = self.db.load_detection_details(inspection_ids)
        
        crack_area = humidity_area = spalling_area = 0.0
        for insp_id, details in details_by_insp.items():
            for d in details:
                if d["class_name"] == "crack":
                    crack_area += d["area_m2"]
                elif d["class_name"] == "humidity":
                    humidity_area += d["area_m2"]
                elif d["class_name"] == "spalling":
                    spalling_area += d["area_m2"]
        
        fig2 = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS["bg_card"])
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(COLORS["bg_card"])
        
        areas = [crack_area, humidity_area, spalling_area]
        labels_pie = ["Grietas", "Humedad", "Desprendimiento"]
        
        filtered = [(l, a, c) for l, a, c in zip(labels_pie, areas, colors_bar) if a > 0]
        
        if filtered:
            labels_f, areas_f, colors_f = zip(*filtered)
            ax2.pie(areas_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold', 'color': COLORS["text_primary"]})
            ax2.axis('equal')
        else:
            ax2.text(0.5, 0.5, "Sin datos de área", ha='center', va='center', fontsize=12, color=COLORS["text_muted"])
            ax2.axis('off')
        
        canvas2 = FigureCanvasTkAgg(fig2, master=chart2_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_module_historial(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="REGISTRO HISTÓRICO", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Historial de Inspecciones", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Consulta todas las inspecciones guardadas en la base de datos", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        content = ctk.CTkFrame(self.content, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)
        
        data = self.db.load_historical_data(1000)
        
        if not data:
            ctk.CTkLabel(content, text="Aún no hay inspecciones guardadas.\nVe a 'Análisis en Campo' para procesar imágenes.", font=(FONT_FAMILY, 14), text_color=COLORS["text_muted"]).pack(pady=50)
            return
        
        df = pd.DataFrame(data)
        
        metrics_frame = ctk.CTkFrame(content, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 15))
        
        total_dets = int(df["total_detections"].sum())
        total_area = float(df["total_area_m2"].sum())
        avg_conf = float(df["avg_confidence"].mean()) if "avg_confidence" in df.columns else 0
        
        for label, value, color in [
            ("Imágenes", str(len(df)), COLORS["primary"]),
            ("Detecciones", str(total_dets), COLORS["crack"]),
            ("Superficie", f"{total_area:.3f} m²", COLORS["accent_gold"]),
            ("Confianza", f"{avg_conf:.1f}%", COLORS["success"]),
        ]:
            card = ctk.CTkFrame(metrics_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            card.pack(side="left", fill="both", expand=True, padx=5)
            
            bar = ctk.CTkFrame(card, height=4, fg_color=color, corner_radius=2)
            bar.pack(fill="x", padx=15, pady=(15, 5))
            
            ctk.CTkLabel(card, text=label.upper(), font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_muted"]).pack(pady=(5, 5))
            ctk.CTkLabel(card, text=value, font=(FONT_FAMILY, 22, "bold"), text_color=color).pack(pady=(0, 15))
        
        table_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        table_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(table_frame, text="Tabla de Inspecciones", font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"]).pack(pady=(12, 8), padx=15, anchor="w")
        
        scroll_frame = ctk.CTkScrollableFrame(table_frame, fg_color="transparent", width=1000)
        scroll_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        header_row = ctk.CTkFrame(scroll_frame, fg_color=COLORS["bg_secondary"], corner_radius=6, height=36)
        header_row.pack(fill="x", pady=(0, 6))
        
        columns = ["Fecha", "Archivo", "Total", "Grietas", "Humedad", "Despr.", "Área (m²)", "Conf. (%)"]
        col_widths = [120, 220, 60, 60, 60, 60, 100, 80]
        
        for i, (col, w) in enumerate(zip(columns, col_widths)):
            ctk.CTkLabel(header_row, text=col, font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_primary"], width=w, anchor="w").pack(side="left", padx=(0 if i == 0 else 2, 2), pady=6)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            row_color = COLORS["bg_surface"] if idx % 2 == 0 else COLORS["bg_card"]
            row_frame = ctk.CTkFrame(scroll_frame, fg_color=row_color, corner_radius=4, height=36)
            row_frame.pack(fill="x", pady=2)
            
            created_at = row.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    created_at = "—"
            
            values = [
                created_at,
                row.get("filename", "—"),
                str(row.get("total_detections", 0)),
                str(row.get("crack_count", 0)),
                str(row.get("humidity_count", 0)),
                str(row.get("spalling_count", 0)),
                f"{row.get('total_area_m2', 0):.4f}",
                f"{row.get('avg_confidence', 0):.1f}"
            ]
            
            for i, (val, w) in enumerate(zip(values, col_widths)):
                ctk.CTkLabel(row_frame, text=val, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], width=w, anchor="w", justify="left").pack(side="left", padx=(0 if i == 0 else 2, 2), pady=4)
        
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(
            btn_frame, text="Exportar Historial Completo (CSV)",
            command=lambda: self.export_historical_csv(df),
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            height=40
        ).pack(side="right")
    
    def export_historical_csv(self, df):
        file_path = filedialog.asksaveasfilename(
            title="Exportar Historial CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"historial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not file_path:
            return
        
        try:
            df.to_csv(file_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
            messagebox.showinfo("Exportación Exitosa", f"CSV guardado en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_module_especificacion(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="DOCUMENTACIÓN", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Especificación Técnica del Sistema", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Arquitectura, parametrización y estrategia de filtrado para sillar volcánico", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        content = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)
        
        model_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        model_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(model_frame, text="CONFIGURACIÓN DEL MODELO", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        model_info = [
            ("Arquitectura:", "YOLOv8 Small"),
            ("Clases:", "3 (crack, humidity, spalling)"),
            ("mAP@50-95 validado:", "63.7%"),
            ("Pesos:", "best.pt (21.47 MB)"),
        ]
        
        for label, value in model_info:
            row = ctk.CTkFrame(model_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=label, font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_muted"], width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(side="left")
        
        env_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        env_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(env_frame, text="ENTORNO DE EJECUCIÓN", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        env_info = [
            ("Framework:", "CustomTkinter (Desktop nativo)"),
            ("Backend ML:", "Ultralytics YOLO v8.x"),
            ("Base de Datos:", "Supabase PostgreSQL (nube)"),
            ("Visualización:", "Matplotlib + OpenCV + Canvas Interactivo"),
            ("Aceleración:", "CUDA si disponible"),
        ]
        
        for label, value in env_info:
            row = ctk.CTkFrame(env_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=label, font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_muted"], width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(side="left")
        
        filters_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        filters_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(filters_frame, text="ESTRATEGIA DE FILTRADO PARA SILLAR VOLCÁNICO", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        filters_info = [
            ("Grietas (Crack)", "Umbral: 0.15", "Alta sensibilidad para fisuras delgadas"),
            ("Humedad (Humidity)", "Umbral: 0.60", "Anti-sombras + reclasifica a grieta si aspect ratio > 4.5"),
            ("Desprendimiento (Spalling)", "Umbral: 0.70", "Filtra textura rugosa natural con umbral alto"),
        ]
        
        for patologia, umbral, justificacion in filters_info:
            row = ctk.CTkFrame(filters_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(row, text=f"- {patologia}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_primary"], anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=umbral, font=(FONT_FAMILY, 9), text_color=COLORS["accent_gold"], anchor="w", padx=10).pack(side="left")
            ctk.CTkLabel(row, text=justificacion, font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"], anchor="w").pack(side="left")
        
        db_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        db_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(db_frame, text="ESQUEMA DE BASE DE DATOS (Supabase)", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        db_text = """
Tabla: inspection_results
id (INTEGER PRIMARY KEY)
filename (TEXT UNIQUE)
crack_count, humidity_count, spalling_count (INTEGER)
total_detections (INTEGER)
total_area_cm2, total_area_m2 (REAL)
avg_confidence (REAL)
created_at (TIMESTAMP)

Tabla: detection_details
id (INTEGER PRIMARY KEY)
inspection_id (FK -> inspection_results.id)
class_name (TEXT)
confidence (REAL)
x1, y1, x2, y2 (INTEGER)
width_px, height_px (INTEGER)
area_cm2, area_m2 (REAL)
gravity_level (TEXT) - NUEVO v6.5
gravity_score (INTEGER) - NUEVO v6.5
"""
        ctk.CTkLabel(db_frame, text=db_text, font=(FONT_FAMILY, 9), text_color=COLORS["text_primary"], justify="left").pack(padx=20, pady=(0, 15), anchor="w")
    
    def on_closing(self):
        self.destroy()

if __name__ == "__main__":
    app = HeritageDetectorDesktop()
    app.mainloop()