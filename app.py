# -*- coding: utf-8 -*-
"""
app.py - Heritage Damage Detector v6.4 (CON VALIDACION DE DUPLICADOS)
Universidad Catolica de Santa Maria - Arequipa, Peru
Modelo: YOLOv8s (3 clases: crack, humidity, spalling)
Metrica validada: mAP@50-95 = 63.7%
Estrategia: Filtrado dinamico por clase + filtro geometrico anti-porosidad
Tiling activo por defecto + filtro anti-sombras
Base de datos: Supabase PostgreSQL via REST (requests directo)
NOVEDAD v6.4:
- Validacion de duplicados ANTES del procesamiento
- Bloqueo de archivos ya existentes en BD
- Feedback inmediato al usuario (1-2 segundos)
============================================================================
"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import torch
import time
import warnings
from pathlib import Path
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# >>> CORRECCION 1: Cargar variables de entorno
load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.use_container_width.')

# --- CONFIGURACION SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://gosywjlocfoettpzafga.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_ZYILG60f1uW9jbVkuOwXaA_bzpwnj3t")

# --- CLIENTE SUPABASE ---
class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
    
    def insert(self, table: str, data: dict | list) -> dict:
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
            return {"data": None, "error": r.text, "success": False, "status": r.status_code}
        except Exception as e:
            return {"data": None, "error": str(e), "success": False, "status": 0}
    
    # >>> CORRECCION 2: Metodo delete corregido para PostgREST
    def delete_inspection(self, inspection_id: int) -> dict:
        """Elimina una inspeccion y sus detalles asociados (CASCADE)"""
        try:
            r = requests.delete(
                f"{self.url}/rest/v1/inspection_results",
                headers=self.headers,
                params={"id": f"eq.{inspection_id}"},
                timeout=10,
            )
            if 200 <= r.status_code < 300:
                return {"success": True, "status": r.status_code}
            return {"success": False, "status": r.status_code, "error": r.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def select(self, table: str, query: str = "*", order: str = None, limit: int = None, filters: dict = None) -> dict:
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
            return {"data": None, "error": r.text, "success": False, "status": r.status_code}
        except Exception as e:
            return {"data": None, "error": str(e), "success": False, "status": 0}
    
    def ping(self) -> tuple[bool, str]:
        try:
            r = requests.get(
                f"{self.url}/rest/v1/inspection_results",
                headers={**self.headers, "Prefer": "count=exact"},
                params={"select": "id", "limit": "1"},
                timeout=8,
            )
            if 200 <= r.status_code < 300:
                return True, "Conectado a Supabase"
            return False, f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            return False, str(e)

@st.cache_resource
def init_supabase():
    client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
    ok, msg = client.ping()
    return client if ok else None, msg

# --- FUNCION DE VALIDACION DE DUPLICADOS ---
def check_duplicate_files(filenames: list) -> dict:
    """Verifica qué archivos ya existen en la base de datos."""
    if supabase_client is None or not filenames:
        return {"duplicates": [], "new_files": filenames}
    
    try:
        # Construir filtro IN para PostgREST (SIN comillas)
        filenames_clean = [f.replace("'", "''") for f in filenames]
        filter_value = f"in.({','.join(filenames_clean)})"
        
        result = supabase_client.select(
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
        st.error(f"Error al verificar duplicados: {str(e)}")
        return {"duplicates": [], "new_files": filenames}
# --- FUNCIONES DE SUPABASE (GUARDADO Y CARGA) ---
def save_to_supabase(results_list, uploaded_files):
    if supabase_client is None:
        return 0, ["Cliente de Supabase no inicializado"]
    
    saved_count = 0
    errors = []
    
    for idx, (result, file_obj) in enumerate(zip(results_list, uploaded_files)):
        # >>> CORRECCION 4: Solo declarar inspection_id UNA vez
        inspection_id = None
        try:
            inspection_payload = {
                "filename": str(file_obj.name),
                "crack_count": int(result["class_counts"].get("crack", 0)),
                "humidity_count": int(result["class_counts"].get("humidity", 0)),
                "spalling_count": int(result["class_counts"].get("spalling", 0)),
                "total_detections": int(result["n_detections"]),
                "total_area_cm2": float(result["total_area_cm2"]),
                "total_area_m2": float(result["total_area_m2"]),
                "avg_confidence": float(result.get("avg_confidence", 0)),
            }
            
            resp = supabase_client.insert("inspection_results", inspection_payload)
            
            if not resp["success"]:
                errors.append(f"{file_obj.name}: {resp.get('error', 'Error desconocido')}")
                continue
            
            # >>> CORRECCION 4: NO resetear inspection_id aqui
            if resp.get("data") and len(resp["data"]) > 0:
                inspection_id = resp["data"][0].get("id")
            
            if not inspection_id:
                errors.append(f"{file_obj.name}: No se recibio ID")
                continue
            
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
                    })
                
                if details_payload:
                    resp2 = supabase_client.insert("detection_details", details_payload)
                    if not resp2["success"]:
                        # >>> CORRECCION 2: Rollback con metodo corregido
                        if inspection_id:
                            supabase_client.delete_inspection(inspection_id)
                        errors.append(f"Detalles {file_obj.name}: {resp2.get('error')} - Se hizo rollback")
                        continue
            
            saved_count += 1
        
        except Exception as e:
            # >>> CORRECCION 2: Rollback en caso de excepcion
            if inspection_id:
                supabase_client.delete_inspection(inspection_id)
            errors.append(f"{file_obj.name}: {str(e)} - Se hizo rollback")
            continue
    
    return saved_count, errors

def load_historical_data():
    if supabase_client is None:
        return None, f"Sin conexion a Supabase: {supabase_msg}"
    
    result = supabase_client.select(
        "inspection_results",
        "*",
        order="created_at.desc",
        limit=100
    )
    
    if result["success"]:
        return result["data"], None
    else:
        return None, result["error"]

# >>> CORRECCION 3: Limite reducido a 10,000
def load_detection_details(inspection_ids: list) -> dict:
    """Carga todos los detalles de deteccion para una lista de inspection_ids."""
    if not inspection_ids or supabase_client is None:
        return {}
    
    ids_str = ", ".join(str(int(id)) for id in inspection_ids if id is not None)
    if not ids_str:
        return {}
    
    result = supabase_client.select(
        "detection_details",
        "inspection_id, class_name, area_m2, area_cm2, confidence, x1, y1, x2, y2, width_px, height_px",
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

# --- EXPORTACION CSV ---
def generate_batch_report(results_list, uploaded_files):
    report_data = []
    for result, file_obj in zip(results_list, uploaded_files):
        report_data.append({
            "Archivo":            file_obj.name,
            "Cracks":             result["class_counts"].get("crack", 0),
            "Humedad":            result["class_counts"].get("humidity", 0),
            "Spalling":           result["class_counts"].get("spalling", 0),
            "Total_Detecciones":  result["n_detections"],
            "Area_cm2":           result["total_area_cm2"],
            "Area_m2":            result["total_area_m2"],
            "Confianza_Promedio": result.get("avg_confidence", 0),
        })
    return pd.DataFrame(report_data)

def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

# --- FILTRO GEOMETRICO + ANTI-SOMBRAS ---
def filtrar_falsos_positivos(detections, image_shape, umbrales_clase):
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

# --- DIBUJADO DE CAJAS ---
def dibujar_cajas_y_segmentos(img_cv2, detections):
    color_map = {"crack": (0, 0, 220), "humidity": (240, 100, 30), "spalling": (0, 140, 235)}
    
    for d in detections:
        color = color_map.get(d["Clase"], (128, 128, 128))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 255, 255), 12)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 8)
        
        conf_porcentaje = d["Confianza"] * 100
        label = f"{d['Clase']} {conf_porcentaje:.1f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)[0]
        label_rect_origin = (x1, max(y1 - label_size[1] - 10, label_size[1] + 10))
        label_text_origin = (x1, max(y1, label_size[1] + 10))
        
        cv2.rectangle(img_cv2, (label_rect_origin[0], label_rect_origin[1] - label_size[1]),
                      (label_rect_origin[0] + label_size[0], label_rect_origin[1]), (255, 255, 255), -1)
        cv2.putText(img_cv2, label, label_text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    
    return img_cv2

# --- FUNCIONES DE TILING ---
def split_image_into_tiles(image_np, tile_size, overlap):
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
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes).astype(np.float32)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
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

def procesar_con_tiling(model, image_pil, class_thresholds, iou_threshold, cm_per_pixel,
                       tile_size=640, overlap=0.2, min_conf=0.05):
    img_np = np.array(image_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_cv2.shape[:2]
    
    tiles = split_image_into_tiles(img_cv2, tile_size, overlap)
    all_detections_raw = []
    inference_log = []
    
    for tile, x_off, y_off in tiles:
        tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        
        try:
            t0 = time.time()
            results = model.predict(
                source=tile_pil, imgsz=tile_size, conf=min_conf, iou=iou_threshold,
                verbose=False, show=False, augment=True,
                agnostic_nms=False, retina_masks=True, half=torch.cuda.is_available(),
            )
            elapsed = (time.time() - t0) * 1000
            n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
            inference_log.append({"tile": (x_off, y_off), "detections": n_dets, "time_ms": round(elapsed, 1)})
            
            if results[0].boxes is None:
                continue
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                
                if conf < class_thresholds.get(cls_name, 0.05):
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += x_off; y1 += y_off; x2 += x_off; y2 += y_off
                
                x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
                
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
            inference_log.append({"tile": (x_off, y_off), "error": str(e), "status": "failed"})
    
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
    
    detections_filtradas = filtrar_falsos_positivos(final_detections, img_cv2.shape, class_thresholds)
    
    for d in detections_filtradas:
        area_cm2 = (d["Ancho_px"] * d["Alto_px"]) * (cm_per_pixel ** 2)
        d["Area_cm2"] = round(area_cm2, 2)
        d["Area_m2"] = round(area_cm2 / 10000, 5)
    
    class_counts = {"crack": 0, "humidity": 0, "spalling": 0}
    total_area_cm2 = 0
    confidences = []
    
    for d in detections_filtradas:
        class_counts[d["Clase"]] += 1
        total_area_cm2 += d["Area_cm2"]
        confidences.append(d["Confianza"])
    
    img_con_cajas = dibujar_cajas_y_segmentos(img_cv2.copy(), detections_filtradas)
    img_rgb = cv2.cvtColor(img_con_cajas, cv2.COLOR_BGR2RGB)
    
    detections_ui = []
    for d in detections_filtradas:
        detections_ui.append({
            "Clase": d["Clase"],
            "Confianza": round(d["Confianza"] * 100, 1),
            "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
            "Ancho_px": d["Ancho_px"], "Alto_px": d["Alto_px"],
            "Area_cm2": d["Area_cm2"], "Area_m2": d["Area_m2"],
        })
    
    return {
        "annotated": Image.fromarray(img_rgb),
        "detections": detections_ui,
        "class_counts": class_counts,
        "total_area_cm2": round(total_area_cm2, 2),
        "total_area_m2": round(total_area_cm2 / 10000, 4),
        "n_detections": len(detections_filtradas),
        "avg_confidence": round(np.mean(confidences) * 100, 1) if confidences else 0,
        "preprocessing": f"Tiling (tile={tile_size}px, overlap={int(overlap*100)}%) + filtro geometrico",
        "inference_log": inference_log,
        "success": True,
    }

# --- FUNCION DE PROCESAMIENTO UNIFICADA ---
def process_image_sillar(model, image, class_thresholds, iou_threshold, cm_per_pixel,
                        use_multiscale=True, use_tiling=False, tile_size=640, overlap=0.2):
    if use_tiling:
        return procesar_con_tiling(model, image, class_thresholds, iou_threshold,
                                  cm_per_pixel, tile_size, overlap)
    else:
        width, height = image.size
        results = None
        preprocessing_info = "Nativa con control de umbrales + filtro geometrico anti-porosidad"
        inference_log = []
        
        imgsz_options = [1536, 1280, 960, 640] if use_multiscale else ([1280] if max(width, height) > 800 else [640])
        
        for imgsz in imgsz_options:
            try:
                t0 = time.time()
                results = model.predict(
                    source=image, imgsz=imgsz, conf=0.05, iou=iou_threshold,
                    verbose=False, show=False, augment=True,
                    agnostic_nms=False, retina_masks=True, half=torch.cuda.is_available(),
                )
                n = len(results[0].boxes) if results[0].boxes is not None else 0
                inference_log.append({"imgsz": imgsz, "time_ms": round((time.time() - t0) * 1000, 1),
                                     "detections": n, "status": "success"})
                if n > 0:
                    break
            except Exception as e:
                inference_log.append({"imgsz": imgsz, "error": str(e), "status": "failed"})
                continue
        
        if results is None or results[0].boxes is None or len(results[0].boxes) == 0:
            return {
                "annotated": image.copy(), "detections": [],
                "class_counts": {"crack": 0, "humidity": 0, "spalling": 0},
                "total_area_cm2": 0, "total_area_m2": 0, "n_detections": 0,
                "avg_confidence": 0, "preprocessing": preprocessing_info,
                "inference_log": inference_log, "success": True,
            }
        
        result = results[0]
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img_cv2.shape[:2]
        detections_raw = []
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            
            if conf < class_thresholds.get(cls_name, 0.25):
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box, h_box = x2 - x1, y2 - y1
            area_cm2 = (w_box * h_box) * (cm_per_pixel ** 2)
            
            detections_raw.append({
                "Clase": cls_name,
                "Confianza": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "Ancho_px": w_box, "Alto_px": h_box,
                "Area_cm2": round(area_cm2, 2),
                "Area_m2": round(area_cm2 / 10000, 5),
            })
        
        detections_filtradas = filtrar_falsos_positivos(detections_raw, img_cv2.shape, class_thresholds)
        
        class_counts_filtrado = {"crack": 0, "humidity": 0, "spalling": 0}
        total_area_cm2_filtrado = 0
        confidences_filtradas = []
        
        for d in detections_filtradas:
            class_counts_filtrado[d["Clase"]] += 1
            total_area_cm2_filtrado += d["Area_cm2"]
            confidences_filtradas.append(d["Confianza"])
        
        img_con_cajas = dibujar_cajas_y_segmentos(img_cv2.copy(), detections_filtradas)
        img_rgb = cv2.cvtColor(img_con_cajas, cv2.COLOR_BGR2RGB)
        
        detections_ui = []
        for d in detections_filtradas:
            detections_ui.append({
                "Clase": d["Clase"],
                "Confianza": round(d["Confianza"] * 100, 1),
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "Ancho_px": d["Ancho_px"], "Alto_px": d["Alto_px"],
                "Area_cm2": d["Area_cm2"], "Area_m2": d["Area_m2"],
            })
        
        return {
            "annotated": Image.fromarray(img_rgb),
            "detections": detections_ui,
            "class_counts": class_counts_filtrado,
            "total_area_cm2": round(total_area_cm2_filtrado, 2),
            "total_area_m2": round(total_area_cm2_filtrado / 10000, 4),
            "n_detections": len(detections_filtradas),
            "avg_confidence": round(np.mean(confidences_filtradas) * 100, 1) if confidences_filtradas else 0,
            "preprocessing": preprocessing_info,
            "inference_log": inference_log,
            "success": True,
        }

# --- COMPONENTES UI ---
def render_badges(class_counts):
    badges = []
    if class_counts.get("crack", 0) > 0:
        badges.append(f'<span class="badge-crack"><span class="badge-dot dot-crack"></span>Grietas: {class_counts["crack"]}</span>')
    if class_counts.get("humidity", 0) > 0:
        badges.append(f'<span class="badge-humidity"><span class="badge-dot dot-humidity"></span>Humedad: {class_counts["humidity"]}</span>')
    if class_counts.get("spalling", 0) > 0:
        badges.append(f'<span class="badge-spalling"><span class="badge-dot dot-spalling"></span>Desprendimiento: {class_counts["spalling"]}</span>')
    
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-empty"><span class="badge-dot dot-ok"></span>Sin patologias detectadas bajo los filtros configurados para sillar.</div>', unsafe_allow_html=True)

def render_detection_table(detections):
    if not detections:
        return
    
    df = pd.DataFrame(detections)
    display_df = df[["Clase", "Confianza", "Area_cm2"]].copy()
    display_df["Confianza"] = display_df["Confianza"].apply(lambda x: f"{x:.1f}%")
    display_df["Area_cm2"] = display_df["Area_cm2"].apply(lambda x: f"{x:.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_debug_info(result):
    with st.expander("Metricas de Inferencia", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Detalle de inferencia")
            for log in result.get("inference_log", []):
                if "imgsz" in log:
                    st.text(f"Escala {log['imgsz']}px: {log['detections']} candidatas ({log['time_ms']}ms)")
                elif "tile" in log:
                    st.text(f"Tile {log['tile']}: {log.get('detections',0)} detecciones ({log.get('time_ms',0)}ms)")
        with col2:
            st.markdown("Consolidado post-filtro")
            st.metric("Detecciones validadas", result["n_detections"])
            st.metric("Confianza media", f"{result.get('avg_confidence', 0):.1f}%")
            st.metric("Preprocesamiento", result.get("preprocessing", "N/A"))

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(
    page_title="Heritage Damage Detector | UCSM",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# >>> CORRECCION 5: Uploader con key dinamica
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- CSS PREMIUM (SIN DUPLICADOS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
--c-primary-950: #061729;
--c-primary-900: #0A2540;
--c-primary-800: #0F2E4F;
--c-primary-700: #1E3A5F;
--c-primary-600: #2D4A72;
--c-primary-500: #3D5A80;
--c-primary-400: #5B7BA5;
--c-primary-300: #8FA8C7;

--c-gold-700: #8B6508;
--c-gold-600: #B8860B;
--c-gold-500: #D4A574;
--c-gold-400: #E4C39A;
--c-gold-300: #F0DFBF;
--c-gold-100: #FBF4E4;

--c-terra-700: #8B3A1F;
--c-terra-600: #C65D3D;
--c-terra-500: #D97A5A;

--c-success-700: #065F46;
--c-success-600: #0F766E;
--c-success-500: #14B8A6;
--c-success-100: #CCFBF1;

--c-warning-700: #92400E;
--c-warning-600: #B45309;
--c-warning-500: #F59E0B;
--c-warning-100: #FEF3C7;

--c-error-700: #991B1B;
--c-error-600: #B91C1C;
--c-error-500: #DC2626;
--c-error-100: #FEE2E2;

--c-crack: #B91C1C;
--c-humidity: #1E40AF;
--c-spalling: #92400E;

--c-neutral-950: #0B1120;
--c-neutral-900: #0F172A;
--c-neutral-800: #1E293B;
--c-neutral-700: #334155;
--c-neutral-600: #475569;
--c-neutral-500: #64748B;
--c-neutral-400: #94A3B8;
--c-neutral-300: #CBD5E1;
--c-neutral-200: #E2E8F0;
--c-neutral-100: #F1F5F9;
--c-neutral-50: #F8FAFC;

--c-bg:        #FAFAF7;
--c-bg-warm:   #F5F2EA;
--c-surface:   #FFFFFF;
--c-border:    #E7E5DF;
--c-border-strong: #D4D0C6;

--shadow-xs:  0 1px 2px rgba(15, 23, 42, 0.04);
--shadow-sm:  0 2px 6px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.03);
--shadow-md:  0 4px 14px rgba(15, 23, 42, 0.06), 0 2px 4px rgba(15, 23, 42, 0.04);
--shadow-lg:  0 12px 28px rgba(15, 23, 42, 0.08), 0 4px 8px rgba(15, 23, 42, 0.04);
--shadow-xl:  0 24px 48px rgba(15, 23, 42, 0.12), 0 8px 16px rgba(15, 23, 42, 0.06);
--shadow-gold: 0 8px 28px rgba(184, 134, 11, 0.18);

--radius-sm: 8px;
--radius-md: 12px;
--radius-lg: 16px;
--radius-xl: 20px;

--ease: cubic-bezier(0.4, 0, 0.2, 1);
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}

.main-header h1,
.main-header .eyebrow {
color: #FFFFFF !important;
-webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stFormSubmitButton"],
div[data-testid="stFormSubmitButton"] > button,
button[data-testid="stBaseButton-primary"] {
color: #FFFFFF !important;
background: linear-gradient(135deg, var(--c-primary-800) 0%, var(--c-primary-700) 60%, var(--c-primary-600) 100%) !important;
}

[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
background: linear-gradient(135deg, #BFDBFE, #93C5FD) !important;
color: #0F172A !important;
box-shadow: var(--shadow-sm) !important;
border: 2px solid #3B82F6 !important;
}

.stFileUploader label,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] > div > div > label {
color: #FFFFFF !important;
font-weight: 600 !important;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
color: #FFFFFF !important;
}

h1, .stMarkdown h1 {
color: #FFFFFF !important;
}

.section-title {
color: var(--c-neutral-700) !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
color: var(--c-neutral-900) !important;
-webkit-font-smoothing: antialiased;
-moz-osx-font-smoothing: grayscale;
}

[data-testid="stAppViewContainer"] {
background:
radial-gradient(ellipse 80% 50% at 50% -20%, rgba(184, 134, 11, 0.04), transparent),
radial-gradient(ellipse 60% 40% at 100% 0%, rgba(30, 58, 95, 0.03), transparent),
linear-gradient(180deg, #FAFAF7 0%, #F5F2EA 100%) !important;
background-attachment: fixed;
}

[data-testid="stAppViewContainer"] > .main {
padding-top: 1.5rem;
}

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
color: var(--c-neutral-900) !important;
letter-spacing: -0.02em;
font-weight: 700;
}

p, span, div, li, label {
color: var(--c-neutral-800);
}

code, pre, .stCodeBlock {
font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
}

@keyframes fadeInUp {
from { opacity: 0; transform: translateY(16px); }
to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
from { opacity: 0; }
to   { opacity: 1; }
}
@keyframes subtleFloat {
0%, 100% { transform: translateY(0); }
50%      { transform: translateY(-3px); }
}
@keyframes gradientShift {
0%, 100% { background-position: 0% 50%; }
50%      { background-position: 100% 50%; }
}
@keyframes shimmer {
0%   { background-position: -200% center; }
100% { background-position: 200% center; }
}
@keyframes glowPulse {
0%, 100% { box-shadow: 0 0 0 0 rgba(184, 134, 11, 0.0); }
50%      { box-shadow: 0 0 24px 2px rgba(184, 134, 11, 0.12); }
}
@keyframes borderGlow {
0%, 100% { border-color: var(--c-border); }
50%      { border-color: var(--c-gold-400); }
}
@keyframes slideReveal {
from { opacity: 0; transform: translateX(-12px); }
to   { opacity: 1; transform: translateX(0); }
}

.main-header {
position: relative;
background:
linear-gradient(135deg,
var(--c-primary-950) 0%,
var(--c-primary-800) 40%,
var(--c-primary-700) 75%,
var(--c-primary-600) 100%);
background-size: 200% 200%;
animation: gradientShift 18s ease infinite, fadeInUp 0.7s var(--ease) both;
color: #FFFFFF;
padding: 2.75rem 3rem 2.5rem;
border-radius: var(--radius-xl);
margin-bottom: 2.5rem;
box-shadow: var(--shadow-xl), inset 0 1px 0 rgba(255,255,255,0.06);
overflow: hidden;
border: 1px solid rgba(212, 165, 116, 0.18);
}
.main-header::before {
content: " ";
position: absolute;
inset: 0;
background:
radial-gradient(circle at 20% 30%, rgba(212, 165, 116, 0.12), transparent 40%),
radial-gradient(circle at 80% 70%, rgba(198, 93, 61, 0.08), transparent 45%);
pointer-events: none;
}
.main-header::after {
content: " ";
position: absolute;
top: 0; left: 0; right: 0;
height: 1px;
background: linear-gradient(90deg, transparent, rgba(212, 165, 116, 0.5), transparent);
}
.main-header .eyebrow {
display: inline-block;
font-family: 'JetBrains Mono', monospace;
font-size: 0.72rem;
font-weight: 500;
letter-spacing: 0.18em;
text-transform: uppercase;
color: var(--c-gold-400);
padding: 0.35rem 0.85rem;
background: rgba(212, 165, 116, 0.1);
border: 1px solid rgba(212, 165, 116, 0.25);
border-radius: 999px;
margin-bottom: 1.1rem;
position: relative;
z-index: 2;
}
.main-header h1 {
margin: 0 0 0.6rem 0;
font-size: 2.25rem;
font-weight: 800;
color: #FFFFFF !important;
letter-spacing: -0.03em;
line-height: 1.15;
position: relative;
z-index: 2;
}
.main-header p {
margin: 0;
color: rgba(241, 245, 249, 0.82) !important;
font-size: 1.02rem;
font-weight: 400;
line-height: 1.55;
max-width: 680px;
position: relative;
z-index: 2;
}

.metric-card {
position: relative;
background: rgba(255, 255, 255, 0.78);
backdrop-filter: blur(10px);
-webkit-backdrop-filter: blur(10px);
padding: 1.6rem 1.5rem 1.4rem;
border-radius: var(--radius-lg);
border: 1px solid var(--c-border);
box-shadow: var(--shadow-sm);
text-align: left;
margin-bottom: 1rem;
transition: all 0.4s var(--ease);
animation: fadeInUp 0.6s var(--ease) both, subtleFloat 7s ease-in-out infinite;
overflow: hidden;
}
.metric-card::before {
content: " ";
position: absolute;
top: 0; left: 0; right: 0;
height: 3px;
background: linear-gradient(90deg, var(--c-primary-700), var(--c-gold-500));
opacity: 0.9;
}
.metric-card:nth-child(1) { animation-delay: 0.05s, 0s; }
.metric-card:nth-child(1)::before { background: linear-gradient(90deg, var(--c-primary-700), var(--c-primary-500)); }
.metric-card:nth-child(2) { animation-delay: 0.15s, 0.5s; }
.metric-card:nth-child(2)::before { background: linear-gradient(90deg, var(--c-gold-600), var(--c-gold-400)); }
.metric-card:nth-child(3) { animation-delay: 0.25s, 1s; }
.metric-card:nth-child(3)::before { background: linear-gradient(90deg, var(--c-terra-600), var(--c-terra-500)); }

.metric-card:hover {
transform: translateY(-4px);
box-shadow: var(--shadow-lg);
border-color: var(--c-gold-400);
}
.metric-card .metric-label {
font-family: 'JetBrains Mono', monospace;
font-size: 0.7rem;
font-weight: 500;
letter-spacing: 0.14em;
text-transform: uppercase;
color: var(--c-neutral-500);
margin: 0 0 0.6rem;
}
.metric-card h3 {
margin: 0;
font-family: 'Plus Jakarta Sans', sans-serif;
font-size: 2.1rem;
font-weight: 800;
color: var(--c-neutral-900) !important;
letter-spacing: -0.03em;
line-height: 1.05;
background: none;
-webkit-text-fill-color: var(--c-neutral-900);
}
.metric-card p {
margin: 0.5rem 0 0;
font-size: 0.88rem;
color: var(--c-neutral-600) !important;
font-weight: 500;
}

.badge-crack, .badge-humidity, .badge-spalling, .badge-empty {
display: inline-flex;
align-items: center;
gap: 0.5rem;
padding: 0.5rem 1.05rem;
border-radius: 999px;
font-family: 'Inter', sans-serif;
font-size: 0.82rem;
font-weight: 600;
margin: 0.25rem 0.3rem;
transition: all 0.3s var(--ease);
animation: fadeIn 0.5s var(--ease) both;
border: 1px solid;
}
.badge-dot {
display: inline-block;
width: 8px;
height: 8px;
border-radius: 50%;
box-shadow: 0 0 0 3px rgba(0,0,0,0.04);
}
.dot-crack     { background: var(--c-crack); box-shadow: 0 0 0 3px rgba(185, 28, 28, 0.15); }
.dot-humidity  { background: var(--c-humidity); box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.15); }
.dot-spalling  { background: var(--c-spalling); box-shadow: 0 0 0 3px rgba(146, 64, 14, 0.15); }
.dot-ok         { background: var(--c-success-600); box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15); }

.badge-crack {
background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
color: var(--c-crack);
border-color: rgba(185, 28, 28, 0.22);
}
.badge-humidity {
background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
color: var(--c-humidity);
border-color: rgba(30, 64, 175, 0.22);
}
.badge-spalling {
background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
color: var(--c-spalling);
border-color: rgba(146, 64, 14, 0.22);
}
.badge-empty {
background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
color: var(--c-success-700);
border-color: rgba(15, 118, 110, 0.22);
font-style: normal;
}
.badge-crack:hover, .badge-humidity:hover, .badge-spalling:hover {
transform: translateY(-2px);
box-shadow: var(--shadow-sm);
}

.alert-info, .alert-success, .alert-warning, .alert-error {
padding: 1.1rem 1.3rem;
border-radius: var(--radius-md);
margin: 1rem 0;
font-size: 0.92rem;
line-height: 1.55;
border: 1px solid;
animation: fadeInUp 0.5s var(--ease) both;
position: relative;
padding-left: 3rem;
}
.alert-info::before, .alert-success::before, .alert-warning::before, .alert-error::before {
content: " ";
position: absolute;
left: 1.1rem;
top: 50%;
transform: translateY(-50%);
width: 10px;
height: 10px;
border-radius: 50%;
}
.alert-info {
background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
border-left: 3px solid var(--c-primary-600);
border-color: rgba(30, 58, 95, 0.15);
color: var(--c-primary-800);
}
.alert-info::before { background: var(--c-primary-600); box-shadow: 0 0 0 4px rgba(30, 58, 95, 0.15); }

.alert-success {
background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
border-left: 3px solid var(--c-success-600);
border-color: rgba(15, 118, 110, 0.15);
color: var(--c-success-700);
}
.alert-success::before { background: var(--c-success-600); box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.15); }

.alert-warning {
background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
border-left: 3px solid var(--c-warning-600);
border-color: rgba(180, 83, 9, 0.15);
color: var(--c-warning-700);
}
.alert-warning::before { background: var(--c-warning-600); box-shadow: 0 0 0 4px rgba(180, 83, 9, 0.15); }

.alert-error {
background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
border-left: 3px solid var(--c-error-600);
border-color: rgba(185, 28, 28, 0.15);
color: var(--c-error-700);
}
.alert-error::before { background: var(--c-error-600); box-shadow: 0 0 0 4px rgba(185, 28, 28, 0.15); }

[data-testid="stSidebar"] {
background: linear-gradient(180deg, #FFFFFF 0%, #FAFAF7 100%) !important;
border-right: 1px solid var(--c-border) !important;
}
[data-testid="stSidebar"] .sidebar-brand {
text-align: center;
padding: 1.8rem 1.2rem 1.5rem;
border-bottom: 1px solid var(--c-border);
margin-bottom: 1.5rem;
position: relative;
}
[data-testid="stSidebar"] .sidebar-brand::after {
content: " ";
position: absolute;
bottom: -1px;
left: 25%;
right: 25%;
height: 2px;
background: linear-gradient(90deg, transparent, var(--c-gold-500), transparent);
}
.sidebar-brand-line1 {
font-family: 'Plus Jakarta Sans', sans-serif;
font-size: 1.55rem;
font-weight: 800;
letter-spacing: 0.04em;
background: linear-gradient(135deg, var(--c-primary-900), var(--c-primary-600));
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
line-height: 1;
}
.sidebar-brand-line2 {
font-family: 'Plus Jakarta Sans', sans-serif;
font-size: 1.55rem;
font-weight: 800;
letter-spacing: 0.04em;
background: linear-gradient(135deg, var(--c-gold-700), var(--c-gold-500));
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
line-height: 1.1;
margin-top: -2px;
}
.sidebar-brand-version {
font-family: 'JetBrains Mono', monospace;
font-size: 0.68rem;
font-weight: 500;
color: var(--c-neutral-500);
letter-spacing: 0.12em;
margin-top: 0.65rem;
text-transform: uppercase;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
font-family: 'Plus Jakarta Sans', sans-serif !important;
font-size: 0.78rem !important;
font-weight: 700 !important;
letter-spacing: 0.14em !important;
text-transform: uppercase !important;
color: var(--c-neutral-700) !important;
margin: 1.8rem 0 0.9rem !important;
padding-bottom: 0.55rem;
border-bottom: 1px solid var(--c-border);
position: relative;
}
[data-testid="stSidebar"] h3::after {
content: " ";
position: absolute;
bottom: -1px;
left: 0;
width: 28px;
height: 2px;
background: var(--c-gold-500);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
color: var(--c-neutral-800) !important;
font-weight: 500;
}

[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stNumberInput > label {
font-size: 0.85rem !important;
font-weight: 600 !important;
color: var(--c-neutral-800) !important;
}

[data-testid="stSidebar"] .stSlider > div > div > div {
background: linear-gradient(90deg, var(--c-primary-700), var(--c-gold-500)) !important;
height: 4px;
border-radius: 2px;
}

[data-testid="stSidebar"] .stRadio > div {
gap: 0.4rem;
background: var(--c-neutral-50);
padding: 0.5rem;
border-radius: var(--radius-md);
border: 1px solid var(--c-border);
}
[data-testid="stSidebar"] .stRadio > div > label {
padding: 0.6rem 0.85rem;
border-radius: var(--radius-sm);
transition: all 0.25s var(--ease);
font-weight: 500;
width: 100%;
font-size: 0.9rem;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
background: rgba(184, 134, 11, 0.08);
}
[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
background: linear-gradient(135deg, var(--c-primary-800), var(--c-primary-700));
color: #FFFFFF !important;
box-shadow: var(--shadow-sm);
}
[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) > div {
color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
display: none;
}

.db-status-ok, .db-status-fail {
padding: 0.8rem 1rem;
border-radius: var(--radius-md);
font-size: 0.82rem;
font-weight: 600;
text-align: center;
margin-top: 1rem;
border: 1px solid;
display: flex;
align-items: center;
justify-content: center;
gap: 0.55rem;
}
.db-status-ok::before, .db-status-fail::before {
content: " ";
width: 8px;
height: 8px;
border-radius: 50%;
animation: glowPulse 2.5s ease-in-out infinite;
}
.db-status-ok {
background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
border-color: rgba(15, 118, 110, 0.25);
color: var(--c-success-700);
}
.db-status-ok::before { background: var(--c-success-600); }
.db-status-fail {
background: linear-gradient(135deg, #FFFBE B, #FEF3C7);
border-color: rgba(180, 83, 9, 0.25);
color: var(--c-warning-700);
}
.db-status-fail::before { background: var(--c-warning-600); }

.filters-info {
background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
border: 1px solid var(--c-border);
border-left: 3px solid var(--c-primary-600);
padding: 1rem 1.15rem;
border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
margin: 1.2rem 0 0.5rem;
font-size: 0.8rem;
line-height: 1.65;
color: var(--c-neutral-800);
}
.filters-info strong {
color: var(--c-primary-800);
font-weight: 700;
letter-spacing: 0.01em;
}
.filters-info code {
background: rgba(30, 58, 95, 0.08);
padding: 0.1rem 0.4rem;
border-radius: 4px;
font-size: 0.78rem;
color: var(--c-primary-800);
}

.stButton > button {
font-family: 'Plus Jakarta Sans', sans-serif !important;
font-weight: 600 !important;
letter-spacing: 0.01em;
border-radius: var(--radius-md) !important;
padding: 0.7rem 1.6rem !important;
transition: all 0.3s var(--ease) !important;
border: 1px solid transparent !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stFormSubmitButton"] {
background: linear-gradient(135deg, var(--c-primary-800) 0%, var(--c-primary-700) 60%, var(--c-primary-600) 100%) !important;
color: #FFFFFF !important;
box-shadow: var(--shadow-sm), inset 0 1px 0 rgba(255,255,255,0.12) !important;
}
.stButton > button[kind="primary"]:hover {
transform: translateY(-2px) !important;
box-shadow: var(--shadow-lg), 0 12px 24px rgba(10, 37, 64, 0.25), inset 0 1px 0 rgba(255,255,255,0.18) !important;
background: linear-gradient(135deg, var(--c-primary-700) 0%, var(--c-primary-600) 100%) !important;
}
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
background: #FFFFFF !important;
color: var(--c-primary-800) !important;
border: 1px solid var(--c-border-strong) !important;
box-shadow: var(--shadow-xs) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
background: var(--c-neutral-50) !important;
border-color: var(--c-gold-500) !important;
color: var(--c-primary-800) !important;
transform: translateY(-1px) !important;
box-shadow: var(--shadow-sm) !important;
}

.stDownloadButton > button {
background: #FFFFFF !important;
color: var(--c-primary-800) !important;
border: 1px solid var(--c-border-strong) !important;
font-weight: 600 !important;
}
.stDownloadButton > button:hover {
background: var(--c-gold-100) !important;
border-color: var(--c-gold-500) !important;
color: var(--c-gold-700) !important;
}

[data-testid="stExpander"] {
background: var(--c-surface);
border: 1px solid var(--c-border);
border-radius: var(--radius-md);
box-shadow: var(--shadow-xs);
}
[data-testid="stExpander"] summary {
font-family: 'Plus Jakarta Sans', sans-serif;
font-weight: 600;
color: var(--c-neutral-800);
padding: 0.9rem 1.1rem;
}
[data-testid="stExpander"] summary:hover {
background: var(--c-neutral-50);
border-radius: var(--radius-md);
}

.stDataFrame, .stTable {
border-radius: var(--radius-md) !important;
overflow: hidden;
border: 1px solid var(--c-border);
box-shadow: var(--shadow-xs);
}
.stDataFrame [data-testid="stDataFrameResizable"] {
border-radius: var(--radius-md);
}

[data-testid="stMetric"] {
background: var(--c-surface);
padding: 1rem 1.2rem;
border-radius: var(--radius-md);
border: 1px solid var(--c-border);
box-shadow: var(--shadow-xs);
}
[data-testid="stMetricLabel"] {
font-family: 'JetBrains Mono', monospace !important;
font-size: 0.72rem !important;
font-weight: 500 !important;
letter-spacing: 0.1em !important;
text-transform: uppercase !important;
color: var(--c-neutral-500) !important;
}
[data-testid="stMetricValue"] {
font-family: 'Plus Jakarta Sans', sans-serif !important;
font-weight: 700 !important;
color: var(--c-neutral-900) !important;
}

hr {
border: none !important;
height: 1px !important;
background: linear-gradient(90deg, transparent, var(--c-border-strong), transparent) !important;
margin: 2rem 0 !important;
}

.image-frame {
position: relative;
background: #FFFFFF;
padding: 1rem;
border-radius: var(--radius-lg);
border: 1px solid var(--c-border);
box-shadow: var(--shadow-md);
animation: fadeIn 0.6s var(--ease) both;
}
.image-frame::before {
content: " ";
position: absolute;
top: -1px; left: -1px; right: -1px;
height: 3px;
background: linear-gradient(90deg, var(--c-primary-700), var(--c-gold-500), var(--c-terra-600));
border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}

.result-card {
background: var(--c-surface);
border: 1px solid var(--c-border);
border-radius: var(--radius-lg);
padding: 1.5rem;
margin-bottom: 1.5rem;
box-shadow: var(--shadow-sm);
animation: fadeInUp 0.6s var(--ease) both;
}
.result-card-header {
display: flex;
align-items: center;
gap: 0.8rem;
padding-bottom: 1rem;
margin-bottom: 1.2rem;
border-bottom: 1px solid var(--c-border);
}
.result-card-header .file-icon {
width: 36px;
height: 36px;
border-radius: var(--radius-sm);
background: linear-gradient(135deg, var(--c-gold-100), var(--c-gold-300));
display: flex;
align-items: center;
justify-content: center;
color: var(--c-gold-700);
font-family: 'JetBrains Mono', monospace;
font-weight: 700;
font-size: 0.8rem;
}
.result-card-header .file-name {
font-family: 'JetBrains Mono', monospace;
font-size: 0.9rem;
font-weight: 600;
color: var(--c-neutral-900);
letter-spacing: -0.01em;
}

.section-title {
font-family: 'Plus Jakarta Sans', sans-serif;
font-size: 0.78rem;
font-weight: 700;
letter-spacing: 0.14em;
text-transform: uppercase;
color: var(--c-neutral-600);
margin: 1.5rem 0 0.9rem;
display: flex;
align-items: center;
gap: 0.65rem;
}
.section-title::before {
content: " ";
width: 3px;
height: 14px;
background: var(--c-gold-500);
border-radius: 2px;
}

.chart-container {
background: var(--c-surface);
border: 1px solid var(--c-border);
border-radius: var(--radius-lg);
padding: 1.5rem;
box-shadow: var(--shadow-xs);
animation: fadeInUp 0.6s var(--ease) both;
}
.chart-container h4 {
font-family: 'Plus Jakarta Sans', sans-serif;
font-size: 1.05rem;
font-weight: 700;
color: var(--c-neutral-900);
margin: 0 0 1.2rem 0;
letter-spacing: -0.01em;
}

.app-footer {
text-align: center;
padding: 3rem 1.5rem 2rem;
margin-top: 4rem;
animation: fadeIn 0.8s var(--ease) both;
}
.footer-divider {
height: 1px;
background: linear-gradient(90deg, transparent, var(--c-gold-500), var(--c-primary-600), transparent);
margin-bottom: 2rem;
border-radius: 999px;
opacity: 0.5;
}
.footer-brand {
font-family: 'Plus Jakarta Sans', sans-serif;
font-size: 1.2rem;
font-weight: 800;
letter-spacing: -0.01em;
background: linear-gradient(135deg, var(--c-primary-800), var(--c-gold-600));
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
.footer-sub {
color: var(--c-neutral-600);
font-size: 0.88rem;
margin-top: 0.4rem;
font-weight: 500;
}
.footer-loc {
color: var(--c-neutral-500);
font-size: 0.82rem;
margin-top: 0.25rem;
}
.footer-tags {
display: flex;
justify-content: center;
gap: 0.55rem;
margin-top: 1.2rem;
flex-wrap: wrap;
}
.footer-tag {
font-family: 'JetBrains Mono', monospace;
background: var(--c-surface);
border: 1px solid var(--c-border);
color: var(--c-neutral-700);
padding: 0.32rem 0.85rem;
border-radius: 999px;
font-size: 0.72rem;
font-weight: 500;
letter-spacing: 0.02em;
transition: all 0.25s var(--ease);
}
.footer-tag:hover {
border-color: var(--c-gold-500);
color: var(--c-gold-700);
transform: translateY(-1px);
}

.stJson {
background: var(--c-neutral-900) !important;
border-radius: var(--radius-md) !important;
border: 1px solid var(--c-neutral-700) !important;
}

[data-testid="stFileUploader"] {
border: 2px dashed var(--c-border-strong) !important;
border-radius: var(--radius-lg) !important;
padding: 2rem !important;
background: var(--c-surface) !important;
transition: all 0.3s var(--ease);
}
[data-testid="stFileUploader"]:hover {
border-color: var(--c-gold-500) !important;
background: var(--c-gold-100) !important;
}

.stCheckbox label {
color: var(--c-neutral-800) !important;
font-weight: 500 !important;
}

.stSpinner > div {
border-top-color: var(--c-gold-500) !important;
}

.stProgress > div > div > div {
background: linear-gradient(90deg, var(--c-primary-700), var(--c-gold-500)) !important;
}

.stAlert {
border-radius: var(--radius-md) !important;
border: 1px solid var(--c-border) !important;
box-shadow: var(--shadow-sm) !important;
}

.spec-table {
background: var(--c-surface);
border: 1px solid var(--c-border);
border-radius: var(--radius-md);
overflow: hidden;
}

@media (max-width: 768px) {
.main-header { padding: 1.8rem 1.5rem; }
.main-header h1 { font-size: 1.6rem; }
.main-header p { font-size: 0.92rem; }
.metric-card h3 { font-size: 1.6rem; }
.metric-card { padding: 1.2rem 1.1rem; }
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--c-neutral-50); }
::-webkit-scrollbar-thumb {
background: var(--c-neutral-300);
border-radius: 5px;
border: 2px solid var(--c-neutral-50);
}
::-webkit-scrollbar-thumb:hover { background: var(--c-gold-500); }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAR SUPABASE ---
supabase_client, supabase_msg = init_supabase()

# --- ESTADO DE SESION ---
if 'all_results' not in st.session_state:
    st.session_state.all_results = []
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'save_attempted' not in st.session_state:
    st.session_state.save_attempted = False
if 'save_result' not in st.session_state:
    st.session_state.save_result = None
if 'fresh_analysis' not in st.session_state:
    st.session_state.fresh_analysis = False

# --- CARGA DEL MODELO ---
@st.cache_resource(show_spinner="Inicializando modelo YOLOv8...")
def load_model(model_path="best.pt"):
    try:
        if not Path(model_path).exists():
            st.error(f"Modelo no encontrado: {model_path}")
            st.info("Coloca best.pt en la misma carpeta que app.py")
            return None
        model = YOLO(model_path)
        _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, imgsz=640)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {type(e).__name__}")
        st.code(str(e))
        return None

model = load_model()
if model is None:
    st.stop()

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-line1">HERITAGE</div>
        <div class="sidebar-brand-line2">DETECTOR</div>
        <div class="sidebar-brand-version">v6.4 · UCSM · Premium Edition</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Modulos",
        ["Analisis en Campo", "Cuadro de Mandos", "Historial", "Especificacion Tecnica"],
        label_visibility="collapsed",
        index=0,
    )
    
    st.markdown("### Calibracion para Sillar")
    st.markdown("<small style='color:var(--c-neutral-600);'>Ajusta segun la porosidad de la piedra</small>", unsafe_allow_html=True)
    
    t_crack = st.slider("Grietas (Crack)", min_value=0.05, max_value=0.90, value=0.15, step=0.01,
                       help="Umbral bajo para capturar fisuras delgadas en sillar")
    t_humidity = st.slider("Humedad (Humidity)", min_value=0.05, max_value=0.90, value=0.60, step=0.01,
                          help="Umbral alto para evitar confusion con sombras y porosidad")
    t_spalling = st.slider("Desprendimiento (Spalling)", min_value=0.05, max_value=0.90, value=0.70, step=0.01,
                          help="Umbral alto para filtrar textura rugosa natural")
    
    class_thresholds = {"crack": t_crack, "humidity": t_humidity, "spalling": t_spalling}
    iou_threshold = st.slider("IoU Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    cm_per_pixel = st.number_input("GSD (cm/pixel)", min_value=0.01, max_value=1.0, value=0.13, step=0.01)
    
    st.markdown("### Micro-inferencia")
    
    use_tiling = st.checkbox("Detectar detalles pequenos (tiling)", value=True,
                            help="Divide la imagen en parches para detectar danos pequenos en tomas lejanas")
    if use_tiling:
        st.warning("Activar el Tiling puede aumentar significativamente el tiempo de procesamiento.")
    
    if use_tiling:
        tile_size = st.slider("Tamano del tile (px)", min_value=320, max_value=1280, value=640, step=64)
        overlap_percent = st.slider("Solapamiento (%)", min_value=0, max_value=50, value=20, step=5)
        overlap = overlap_percent / 100.0
    else:
        tile_size = 640
        overlap = 0.2
    
    multiscale = st.checkbox("Multi-scale inference", value=True)
    debug_mode = st.checkbox("Mostrar diagnostico tecnico", value=False)
    
    st.markdown("""
    <div class="filters-info">
        <strong>Filtros activos</strong><br>
        &bull; Anti-sombras (humedad grande y alargada)<br>
        &bull; Humedad delgada reclasifica a grieta<br>
        &bull; Spalling requiere confianza &gt; <code>70%</code>
    </div>
    """, unsafe_allow_html=True)
    
    if supabase_client:
        st.markdown('<div class="db-status-ok">Base de Datos conectada</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="db-status-fail">BD en modo offline</div>', unsafe_allow_html=True)

# --- MODULO 1: ANALISIS EN CAMPO ---
if page == "Analisis en Campo":
    st.markdown("""
    <div class="main-header">
        <span class="eyebrow">Inspeccion Estructural</span>
        <h1>Analisis de Sillar Volcanico</h1>
        <p>Santuario de San Agustin, Arequipa. Control de danos patrimoniales con filtrado geometrico anti-porosidad, anti-sombras y micro-inferencia por tiling.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Cargar imagenes del monumento (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"uploader_main_{st.session_state.uploader_key}"
    )
    
    if uploaded_files:
        # VALIDACION DE DUPLICADOS INMEDIATA
        filenames = [f.name for f in uploaded_files]
        
        with st.spinner("⏳ Verificando archivos en la base de datos..."):
            validation = check_duplicate_files(filenames)
        
        duplicates = validation["duplicates"]
        new_files = validation["new_files"]
        
        if duplicates:
            st.error("""
            ### ⛔ ARCHIVOS DUPLICADOS DETECTADOS
            
            Los siguientes archivos **YA EXISTEN** en la base de datos y no pueden ser procesados nuevamente:
            """)
            
            for dup in duplicates:
                st.error(f"❌ **{dup}**")
            
            if new_files:
                st.warning(f"""
                ##### ✅ Archivos nuevos disponibles: {len(new_files)}
                
                Los siguientes archivos son nuevos y pueden ser procesados:
                """)
                for nf in new_files:
                    st.success(f"✓ {nf}")
                
                st.info("💡 **Recomendacion:** Elimina los archivos duplicados del selector y deja solo los nuevos antes de procesar.")
            else:
                st.error("""
                ##### ⚠️ Todos los archivos seleccionados son duplicados
                
                Por favor, **selecciona otros archivos** que no hayan sido procesados anteriormente.
                """)
            
            st.stop()
        
        else:
            st.success(f"✅ **Validacion completada:** {len(uploaded_files)} archivo(s) nuevos listos para procesar")
    
    if st.session_state.all_results and not uploaded_files:
        st.markdown(f"""
        <div class="alert-info">
            <strong>📋 Resultados en memoria:</strong> 
            Tienes {len(st.session_state.all_results)} imagen(es) procesada(s) pendientes de guardar.
            Puedes guardarlas en Supabase o procesar nuevas imágenes.
        </div>
        """, unsafe_allow_html=True)
        
        for idx, (result, file_obj) in enumerate(zip(st.session_state.all_results, st.session_state.uploaded_files_list)):
            with st.expander(f"📷 {file_obj.name} - {result['n_detections']} detecciones", expanded=False):
                col_img, col_info = st.columns([1.4, 1])
                with col_img:
                    st.image(result["annotated"], caption="Imagen procesada", use_container_width=True)
                with col_info:
                    st.markdown(f"**Detecciones:** {result['n_detections']}")
                    st.markdown(f"**Área total:** {result['total_area_m2']:.4f} m²")
                    render_badges(result["class_counts"])
    
    if uploaded_files and st.button("INICIAR PROCESAMIENTO CON FILTRO SILLAR", type="primary", key="btn_process"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Procesando imagenes con YOLO..."):
            st.session_state.all_results = []
            st.session_state.uploaded_files_list = []
            st.session_state.fresh_analysis = False
            
            for idx, uploaded_file in enumerate(uploaded_files):
                progress_bar.progress((idx + 1) / len(uploaded_files))
                status_text.text(f"Procesando {idx + 1} de {len(uploaded_files)}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file).convert("RGB")
                result = process_image_sillar(
                    model, image, class_thresholds, iou_threshold, cm_per_pixel,
                    use_multiscale=multiscale,
                    use_tiling=use_tiling,
                    tile_size=tile_size if use_tiling else 640,
                    overlap=overlap if use_tiling else 0.2
                )
                st.session_state.all_results.append(result)
                st.session_state.uploaded_files_list.append(uploaded_file)
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-card-header">
                        <div class="file-icon">IMG</div>
                        <div class="file-name">{uploaded_file.name}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                col_img, col_info = st.columns([1.4, 1])
                with col_img:
                    if result["n_detections"] > 0:
                        st.success(f"Validadas {result['n_detections']} regiones tras filtro geometrico")
                    else:
                        st.info("Sin patologias detectadas bajo los filtros configurados")
                    try:
                        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                        st.image(result["annotated"], caption="Mapeo de danos estructurales", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception:
                        st.error("Error al mostrar la imagen procesada.")
                
                with col_info:
                    if debug_mode:
                        render_debug_info(result)
                        st.markdown("---")
                    
                    st.markdown('<div class="section-title">Distribucion por tipo</div>', unsafe_allow_html=True)
                    render_badges(result["class_counts"])
                    
                    st.markdown('<div class="section-title">Area afectada</div>', unsafe_allow_html=True)
                    st.info(f"**Area validada:** {result['total_area_cm2']:.2f} cm2 (~{result['total_area_m2']:.4f} m2)")
                    
                    if result["detections"]:
                        st.markdown('<div class="section-title">Detalle de detecciones</div>', unsafe_allow_html=True)
                        render_detection_table(result["detections"])
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.fresh_analysis = True
            st.success("Procesamiento de lote completado.")
    
    st.markdown("---")
    st.markdown('<div class="section-title">Zona de Guardado y Diagnostico</div>', unsafe_allow_html=True)
    
    if st.session_state.all_results:
        st.info(f"**Estado de Memoria:** {len(st.session_state.all_results)} resultados procesados | {len(st.session_state.uploaded_files_list)} archivos cargados | Frescos: {'Si' if st.session_state.fresh_analysis else 'No'}")
        
        if supabase_client is None:
            st.markdown(f'<div class="alert-warning"><strong>Sin conexion a Supabase.</strong><br>{supabase_msg}</div>', unsafe_allow_html=True)
        else:
            if not st.session_state.fresh_analysis:
                st.markdown(
                    '<div class="alert-warning">'
                    '<strong>Datos no guardables.</strong><br>'
                    'Estos datos fueron cargados desde la base de datos o ya fueron guardados previamente. '
                    'Solo se pueden guardar analisis nuevos realizados en esta sesion. '
                    'Ve a <strong>Analisis en Campo</strong>, procesa nuevas imagenes y luego podras guardarlas.'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button("Guardar automaticamente en Supabase", type="primary", key="btn_save_auto"):
                    with st.spinner("Guardando inspecciones..."):
                        saved_count, errors = save_to_supabase(
                            st.session_state.all_results,
                            st.session_state.uploaded_files_list,
                        )
                    
                    if saved_count > 0:
                        st.success(f"Exito: {saved_count} inspecciones guardadas.")
                        st.balloons()
                        
                        st.session_state.all_results = []
                        st.session_state.uploaded_files_list = []
                        st.session_state.fresh_analysis = False
                        st.session_state.save_attempted = True
                        st.session_state.uploader_key += 1
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.warning("No se guardo ninguna inspeccion automaticamente.")
                    
                    if errors:
                        with st.expander("Ver errores detallados", expanded=True):
                            for err in errors:
                                st.error(f"{err}")
        
        st.markdown("---")
    
    if st.session_state.all_results:
        st.markdown('<div class="section-title">Exportar reporte</div>', unsafe_allow_html=True)
        report_df = generate_batch_report(
            st.session_state.all_results,
            st.session_state.uploaded_files_list,
        )
        with st.expander("Vista previa del reporte", expanded=False):
            st.dataframe(
                report_df.style.format({"Area_m2": "{:.4f}", "Confianza_Promedio": "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="Descargar CSV",
                data=convert_df_to_csv(report_df),
                file_name=f"reporte_sillar_filtrado_{timestamp}.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.info(f"**Resumen:** {len(st.session_state.uploaded_files_list)} imagenes | {report_df['Total_Detecciones'].sum()} detecciones | {report_df['Area_m2'].sum():.4f} m2 afectados")

# --- MODULO 2: CUADRO DE MANDOS ---
elif page == "Cuadro de Mandos":
    st.markdown("""
    <div class="main-header">
        <span class="eyebrow">Dashboard Ejecutivo</span>
        <h1>Cuadro de Mandos Consolidado</h1>
        <p>Visualizacion estadistica de danos en el patrimonio arquitectonico. Metricas clave, distribucion por tipologia y superficie afectada.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.all_results:
        if supabase_client:
            st.markdown('<div class="alert-info">Cargando datos historicos desde la base de datos...</div>', unsafe_allow_html=True)
            with st.spinner("Consultando Supabase..."):
                data, err = load_historical_data()
            
            if err or not data or len(data) == 0:
                st.markdown('<div class="alert-warning">No hay datos procesados. Ejecute primero <strong>Analisis en Campo</strong> o guarde resultados en Supabase.</div>', unsafe_allow_html=True)
                st.stop()
            else:
                st.session_state.all_results = []
                st.session_state.uploaded_files_list = []
                
                inspections_to_load = data[:100]
                
                for row in inspections_to_load:
                    st.session_state.all_results.append({
                        "class_counts": {
                            "crack": row.get("crack_count", 0),
                            "humidity": row.get("humidity_count", 0),
                            "spalling": row.get("spalling_count", 0),
                        },
                        "total_area_m2": float(row.get("total_area_m2", 0)),
                        "total_area_cm2": float(row.get("total_area_cm2", 0)),
                        "n_detections": row.get("total_detections", 0),
                        "avg_confidence": float(row.get("avg_confidence", 0)),
                        "detections": [],
                        "inspection_id": row.get("id"),
                    })
                    st.session_state.uploaded_files_list.append(
                        type('obj', (object,), {'name': row.get("filename", "N/A")})
                    )
                
                st.markdown('<div class="alert-info">Cargando detalles de detecciones para graficos de area...</div>', unsafe_allow_html=True)
                
                inspection_ids = [r.get("inspection_id") for r in st.session_state.all_results if r.get("inspection_id")]
                
                if inspection_ids:
                    with st.spinner("Consultando detalles de detecciones..."):
                        details_by_inspection = load_detection_details(inspection_ids)
                    
                    if details_by_inspection:
                        for result in st.session_state.all_results:
                            insp_id = result.get("inspection_id")
                            if insp_id and insp_id in details_by_inspection:
                                details = details_by_inspection[insp_id]
                                result["detections"] = [
                                    {
                                        "Clase": str(d.get("class_name", "")),
                                        "Area_m2": float(d.get("area_m2", 0)),
                                        "Area_cm2": float(d.get("area_cm2", 0)),
                                        "Confianza": float(d.get("confidence", 0)),
                                    }
                                    for d in details
                                ]
                        
                        total_details_loaded = sum(len(r["detections"]) for r in st.session_state.all_results)
                        st.success(f"Cargadas {len(data)} inspecciones con {total_details_loaded} detalles de detecciones desde la base de datos")
                    else:
                        st.warning("No se encontraron detalles de detecciones. El grafico de areas no estara disponible.")
                else:
                    st.warning("No se pudieron obtener los IDs de inspeccion.")
                
                st.session_state.fresh_analysis = False
        else:
            st.markdown('<div class="alert-info">No hay datos procesados. Ejecute primero <strong>Analisis en Campo</strong>.</div>', unsafe_allow_html=True)
            st.stop()
    
    total_cracks = sum(r["class_counts"].get("crack", 0) for r in st.session_state.all_results)
    total_humidity = sum(r["class_counts"].get("humidity", 0) for r in st.session_state.all_results)
    total_spalling = sum(r["class_counts"].get("spalling", 0) for r in st.session_state.all_results)
    total_detections = total_cracks + total_humidity + total_spalling
    total_area_m2 = sum(r["total_area_m2"] for r in st.session_state.all_results)
    
    if st.session_state.all_results:
        critical_img = max(st.session_state.all_results, key=lambda x: x["n_detections"])
        critical_idx = st.session_state.all_results.index(critical_img)
        critical_name = st.session_state.uploaded_files_list[critical_idx].name if critical_idx < len(st.session_state.uploaded_files_list) else "N/A"
    else:
        critical_name = "N/A"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Danos</div><h3>{total_detections}</h3><p>Danos validados en el lote</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Superficie</div><h3>{total_area_m2:.4f} m2</h3><p>Area total afectada</p></div>', unsafe_allow_html=True)
    with col3:
        dn = critical_name[:22] + "..." if len(critical_name) > 22 else critical_name
        st.markdown(f'<div class="metric-card"><div class="metric-label">Severidad Maxima</div><h3 style="font-size:1.35rem;">{dn}</h3><p>Imagen con mayor severidad</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="chart-container"><h4>Distribucion de danos por tipo</h4>', unsafe_allow_html=True)
        fig_bar = px.bar(
            pd.DataFrame({"Tipo": ["Grietas", "Humedad", "Desprendimiento"],
                         "Cantidad": [total_cracks, total_humidity, total_spalling]}),
            x="Tipo", y="Cantidad", color="Tipo",
            color_discrete_map={"Grietas": "#B91C1C", "Humedad": "#1E40AF", "Desprendimiento": "#92400E"},
            text="Cantidad", height=360, template="plotly_white",
        )
        fig_bar.update_traces(textposition="outside", marker_line_width=0,
                             textfont=dict(family="Plus Jakarta Sans", size=13, color="#0F172A"))
        fig_bar.update_layout(
            xaxis_title="", yaxis_title="Cantidad", showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(family="Inter", color="#334155"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="chart-container"><h4>Distribucion de area afectada (m2)</h4>', unsafe_allow_html=True)
        crack_area = humidity_area = spalling_area = 0.0
        for r in st.session_state.all_results:
            for d in r.get("detections", []):
                clase = d.get("Clase", "")
                area = float(d.get("Area_m2", 0))
                if clase == "crack":
                    crack_area += area
                elif clase == "humidity":
                    humidity_area += area
                elif clase == "spalling":
                    spalling_area += area
        
        area_data = pd.DataFrame({"Tipo": ["Grietas", "Humedad", "Desprendimiento"],
                                 "Area_m2": [crack_area, humidity_area, spalling_area]})
        area_data = area_data[area_data["Area_m2"] > 0]
        
        if len(area_data) > 0:
            fig_pie = px.pie(area_data, values="Area_m2", names="Tipo", hole=0.52, height=360,
                            color="Tipo",
                            color_discrete_map={"Grietas": "#B91C1C", "Humedad": "#1E40AF", "Desprendimiento": "#92400E"},
                            template="plotly_white")
            fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                 textfont=dict(family="Plus Jakarta Sans", size=12, color="#FFFFFF"))
            fig_pie.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                font=dict(family="Inter", color="#334155"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=40, l=20, r=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No hay datos de area para graficar. Los detalles de detecciones no estan disponibles para estos datos.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-title">Resumen por imagen procesada</div>', unsafe_allow_html=True)
    summary_df = generate_batch_report(st.session_state.all_results, st.session_state.uploaded_files_list)
    st.dataframe(summary_df.style.format({"Area_m2": "{:.4f}", "Confianza_Promedio": "{:.1f}%"}),
                use_container_width=True, hide_index=True)

# --- MODULO 3: HISTORIAL ---
elif page == "Historial":
    st.markdown("""
    <div class="main-header">
        <span class="eyebrow">Registro Historico</span>
        <h1>Historial de Inspecciones</h1>
        <p>Consulta todas las inspecciones guardadas en Supabase. Analisis longitudinal de la evolucion de danos en el patrimonio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if supabase_client is None:
        st.markdown(f'<div class="alert-warning"><strong>Sin conexion a Supabase.</strong><br>{supabase_msg}</div>', unsafe_allow_html=True)
    else:
        if st.button("Cargar historial desde la base de datos", type="primary"):
            with st.spinner("Consultando Supabase..."):
                data, err = load_historical_data()
            
            if err:
                st.error(f"Error al cargar datos: {err}")
                st.info("Verifica que las tablas 'inspection_results' y 'detection_details' existan en Supabase.")
            elif not data or len(data) == 0:
                st.info("Aun no hay inspecciones guardadas en la base de datos.")
                st.markdown("""
                **Como guardar inspecciones:**
                
                1. Ve a la pestana **Analisis en Campo**
                2. Carga algunas imagenes del sillar
                3. Procesa las imagenes con el filtro activado
                4. Haz clic en **Guardar resultados en Supabase**
                """)
            else:
                df = pd.DataFrame(data)
                st.success(f"Se cargaron {len(df)} inspecciones correctamente.")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Imagenes</div><h3>{len(df)}</h3><p>Total procesadas</p></div>', unsafe_allow_html=True)
                with col2:
                    total_dets = int(df["total_detections"].sum()) if "total_detections" in df.columns else 0
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Detecciones</div><h3>{total_dets}</h3><p>Acumulado global</p></div>', unsafe_allow_html=True)
                with col3:
                    total_area = float(df["total_area_m2"].sum()) if "total_area_m2" in df.columns else 0.0
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Superficie</div><h3>{total_area:.3f} m2</h3><p>Area total afectada</p></div>', unsafe_allow_html=True)
                with col4:
                    avg_conf = float(df["avg_confidence"].mean()) if "avg_confidence" in df.columns else 0.0
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Confianza</div><h3>{avg_conf:.1f}%</h3><p>Promedio global</p></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown('<div class="section-title">Tabla de Inspecciones</div>', unsafe_allow_html=True)
                
                display_columns = ["created_at", "filename", "total_detections",
                                  "crack_count", "humidity_count", "spalling_count",
                                  "total_area_m2", "avg_confidence"]
                
                existing_cols = [col for col in display_columns if col in df.columns]
                display_df = df[existing_cols].copy()
                
                column_renames = {
                    "created_at": "Fecha",
                    "filename": "Archivo",
                    "total_detections": "Total",
                    "crack_count": "Grietas",
                    "humidity_count": "Humedad",
                    "spalling_count": "Desprendimiento",
                    "total_area_m2": "Area (m2)",
                    "avg_confidence": "Confianza (%)"
                }
                display_df = display_df.rename(columns={k: v for k, v in column_renames.items() if k in display_df.columns})
                
                if "Fecha" in display_df.columns:
                    try:
                        fechas_dt = pd.to_datetime(display_df["Fecha"], utc=True)
                        fechas_lima = fechas_dt - pd.Timedelta(hours=5)
                        display_df["Fecha"] = fechas_lima.dt.strftime("%Y-%m-%d %H:%M")
                    except Exception as e:
                        st.warning(f"⚠️ Error al convertir fechas: {e}. Mostrando hora original.")
                        display_df["Confianza (%)"] = display_df["Confianza (%)"].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                if len(df) > 1 and "created_at" in df.columns:
                    st.markdown("---")
                    st.markdown('<div class="section-title">Tendencia Temporal de Danos</div>', unsafe_allow_html=True)
                    
                    df["dia"] = pd.to_datetime(df["created_at"]).dt.date
                    daily = df.groupby("dia").agg({
                        "crack_count": "sum" if "crack_count" in df.columns else "count",
                        "humidity_count": "sum" if "humidity_count" in df.columns else "count",
                        "spalling_count": "sum" if "spalling_count" in df.columns else "count",
                    }).reset_index()
                    
                    damage_cols = []
                    if "crack_count" in daily.columns:
                        damage_cols.append(("Grietas", "crack_count", "#B91C1C"))
                    if "humidity_count" in daily.columns:
                        damage_cols.append(("Humedad", "humidity_count", "#1E40AF"))
                    if "spalling_count" in daily.columns:
                        damage_cols.append(("Desprendimiento", "spalling_count", "#92400E"))
                    
                    if damage_cols:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        fig_line = px.line()
                        for name, col, color in damage_cols:
                            fig_line.add_scatter(x=daily["dia"], y=daily[col],
                                               mode='lines+markers', name=name,
                                               line=dict(color=color, width=2.5),
                                               marker=dict(size=8, color=color))
                        
                        fig_line.update_layout(
                            title="Evolucion de danos por fecha",
                            xaxis_title="Fecha",
                            yaxis_title="Numero de Detecciones",
                            legend_title="Tipo de Dano",
                            height=400,
                            template="plotly_white",
                            hovermode='x unified',
                            font=dict(family="Inter", color="#334155"),
                            title_font=dict(family="Plus Jakarta Sans", size=16, color="#0F172A"),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_line, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                csv_data = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                st.download_button(
                    label="Descargar historial completo (CSV)",
                    data=csv_data,
                    file_name=f"historial_supabase_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("Haz clic en el boton 'Cargar historial' para ver los datos guardados en Supabase.")

# --- MODULO 4: ESPECIFICACION TECNICA ---
elif page == "Especificacion Tecnica":
    st.markdown("""
    <div class="main-header">
        <span class="eyebrow">Documentacion</span>
        <h1>Especificacion Tecnica del Sistema</h1>
        <p>Documentacion tecnica para sustentacion academica. Arquitectura, parametrizacion y estrategia de filtrado para sillar volcanico.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Configuracion del modelo</div>', unsafe_allow_html=True)
        st.json({
            "Arquitectura": "YOLOv8 Small",
            "Clases": 3,
            "Mapeo": {0: "crack (grieta)", 1: "humidity (humedad)", 2: "spalling (desprendimiento)"},
            "mAP@50-95 validado": 0.637,
            "Pesos": "best.pt (21.47 MB)",
        })
    with col2:
        st.markdown('<div class="section-title">Entorno de ejecucion</div>', unsafe_allow_html=True)
        st.json({
            "Framework": "Streamlit 1.x",
            "Backend": "Ultralytics YOLO v8.x",
            "Base de Datos": "Supabase PostgreSQL (REST directo)",
            "Visualizacion": "Plotly Express",
            "Procesamiento": "OpenCV, NumPy, Pandas",
            "Aceleracion": "CUDA si disponible",
        })
    
    st.markdown("---")
    st.markdown('<div class="section-title">Estrategia de filtrado para Sillar Volcanico</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Patologia": ["Grietas (Crack)", "Humedad (Humidity)", "Desprendimiento (Spalling)"],
        "Umbral recomendado": [0.15, 0.60, 0.70],
        "Filtro geometrico": [
            "Sin filtro adicional - alta sensibilidad para fisuras delgadas",
            "Reclasifica a grieta si relacion aspecto > 4.5. Filtro anti-sombras para cajas grandes y alargadas.",
            "Descarta detecciones con confianza < 70% y areas muy pequenas (<0.2% de la imagen)",
        ],
        "Justificacion para Sillar": [
            "El sillar tiene fracturas naturales finas que deben detectarse",
            "La porosidad crea sombras alargadas que el modelo confunde con humedad",
            "La textura rugosa genera falsos positivos que se eliminan con umbral alto",
        ],
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown('<div class="section-title">Base de datos Supabase</div>', unsafe_allow_html=True)
    st.markdown("""
    **Tabla `inspection_results`** — una fila por imagen procesada
    
    **Tabla `detection_details`** — una fila por cada bounding box detectado
    
    **Conexion via REST HTTP directo** (sin SDK, mas estable entre versiones)
    
    **Almacenamiento persistente en la nube** — accesible desde cualquier equipo
    """)
    
    st.markdown("---")
    st.markdown('<div class="section-title">Referencias</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Ultralytics YOLOv8:** https://docs.ultralytics.com
    * **Roboflow:** https://roboflow.com
    * **Streamlit:** https://streamlit.io
    * **Supabase:** https://supabase.com/docs/guides/api
    * **Patrimonio Arequipa:** Tecnica constructiva en sillar volcanico
    """)

# --- FOOTER ---
st.markdown("""
<div class="app-footer">
    <div class="footer-divider"></div>
    <div class="footer-brand">Heritage Damage Detector v6.4</div>
    <div class="footer-sub">Proyecto de Investigacion &middot; Universidad Catolica de Santa Maria</div>
    <div class="footer-loc">Santuario de San Agustin &middot; Arequipa, Peru</div>
    <div class="footer-tags">
        <span class="footer-tag">YOLOv8s</span>
        <span class="footer-tag">Streamlit</span>
        <span class="footer-tag">mAP@50-95: 63.7%</span>
        <span class="footer-tag">Supabase</span>
        <span class="footer-tag">Premium Edition</span>
    </div>
</div>
""", unsafe_allow_html=True)