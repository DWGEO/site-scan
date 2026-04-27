from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator, model_validator
from openai import OpenAI
from dotenv import load_dotenv

import os
import json
import math
import requests
import re
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import quote

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
    Image,
    KeepTogether,
)
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")
MAPBOX_BASE_URL = "https://api.mapbox.com/styles/v1/mapbox"
MAPBOX_GEOCODE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places"

QLD_IMAGE_SERVER = (
    "https://spatial-img.information.qld.gov.au/arcgis/rest/services/"
    "TimeSeries/AerialOrtho_AllUsers/ImageServer"
)

QLD_SURFACE_GEOLOGY_LAYER_URL = (
    "https://spatial-gis.information.qld.gov.au/arcgis/rest/services/"
    "GeoscientificInformation/GeologyDetailed/MapServer/15"
)
NSW_SEAMLESS_GEOLOGY_LAYER_URL = (
    "https://gs-seamless.geoscience.nsw.gov.au/arcgis/rest/services/"
    "Geology/Geology_100K/MapServer/0"
)

REQUEST_TIMEOUT = 25
POLYGON_PAD_M = 8
CONTEXT_PAD_M = 35
WIDE_CONTEXT_PAD_M = 90

MAX_INITIAL_SCENES = 5
MAX_FOLLOWUP_SCENES = 8
MAX_AI_IMAGES = 14

REPORTS_DIR = "generated_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

DWGEO_LOGO_PATH = os.environ.get("DWGEO_LOGO_PATH", os.path.join("static", "dwgeo-logo.png")).strip()
SITECLASS_LOGO_PATH = os.environ.get("SITECLASS_LOGO_PATH", os.path.join("static", "siteclassonline-logo.png")).strip()
REPORT_FONT_REGULAR = "Helvetica"
REPORT_FONT_BOLD = "Helvetica-Bold"
REPORT_FONT_SEMIBOLD = "Helvetica-Bold"

def register_report_fonts():
    global REPORT_FONT_REGULAR, REPORT_FONT_BOLD, REPORT_FONT_SEMIBOLD

    candidates = [
        (
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/seguisb.ttf",
        ),
        (
            "C:/Windows/Fonts/aptos.ttf",
            "C:/Windows/Fonts/aptos-bold.ttf",
            "C:/Windows/Fonts/aptos-semibold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
    ]

    for regular_path, bold_path, semibold_path in candidates:
        if all(os.path.exists(p) for p in [regular_path, bold_path, semibold_path]):
            try:
                pdfmetrics.registerFont(TTFont("ReportSans", regular_path))
                pdfmetrics.registerFont(TTFont("ReportSansBold", bold_path))
                pdfmetrics.registerFont(TTFont("ReportSansSemiBold", semibold_path))
                REPORT_FONT_REGULAR = "ReportSans"
                REPORT_FONT_BOLD = "ReportSansBold"
                REPORT_FONT_SEMIBOLD = "ReportSansSemiBold"
                return
            except Exception:
                pass

register_report_fonts()


class SiteRequest(BaseModel):
    address: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    polygon: Optional[Union[List[List[float]], str]] = None
    force_geocode: bool = False
    chip_size_m: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def map_frontend_payload(cls, values):
        if not isinstance(values, dict):
            return values

        data = dict(values)

        # Map frontend/localStorage keys to backend keys
        if not data.get("address") and data.get("selectedAddress"):
            data["address"] = data.get("selectedAddress", "")

        if data.get("lat") is None or data.get("lng") is None:
            centroid = (
                data.get("lotCentroid")
                or data.get("lot_centroid")
                or data.get("centroid")
            )

            if centroid:
                try:
                    if isinstance(centroid, str):
                        centroid = json.loads(centroid)

                    if isinstance(centroid, list) and len(centroid) >= 2:
                        # Stored as [lng, lat]
                        data["lng"] = float(centroid[0])
                        data["lat"] = float(centroid[1])
                except Exception:
                    pass

        if not data.get("polygon"):
            polygon_candidate = (
                data.get("lotPolygon")
                or data.get("lot_polygon")
                or data.get("lotGeojson")
                or data.get("lot_geojson")
                or data.get("polygon")
            )

            if polygon_candidate:
                try:
                    if isinstance(polygon_candidate, str):
                        parsed = json.loads(polygon_candidate)
                    else:
                        parsed = polygon_candidate

                    # Full GeoJSON feature
                    if isinstance(parsed, dict):
                        geometry = parsed.get("geometry", {})
                        if geometry.get("type") == "Polygon":
                            coords = geometry.get("coordinates", [])
                            if coords and isinstance(coords[0], list):
                                data["polygon"] = coords[0]

                    # Already a polygon ring [[lng, lat], ...]
                    elif (
                        isinstance(parsed, list)
                        and parsed
                        and isinstance(parsed[0], list)
                        and len(parsed[0]) == 2
                    ):
                        data["polygon"] = parsed

                    # Polygon coordinates [[[lng, lat], ...]]
                    elif (
                        isinstance(parsed, list)
                        and parsed
                        and isinstance(parsed[0], list)
                        and parsed[0]
                        and isinstance(parsed[0][0], list)
                    ):
                        data["polygon"] = parsed[0]

                except Exception:
                    pass

        if not data.get("address"):
            data["address"] = ""

        return data

    @field_validator("polygon", mode="before")
    @classmethod
    def parse_polygon_if_string(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception as exc:
                raise ValueError("Polygon string could not be parsed as JSON.") from exc
        return value

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if value is None:
            return value

        if len(value) < 3:
            raise ValueError("Polygon must contain at least 3 coordinate pairs.")

        for pt in value:
            if not isinstance(pt, list) or len(pt) != 2:
                raise ValueError("Each polygon coordinate must be [lng, lat].")

            lng, lat = pt
            if not isinstance(lng, (int, float)) or not isinstance(lat, (int, float)):
                raise ValueError("Polygon coordinates must be numeric.")

        return value
    

@app.get("/")
def root():
    return {"message": "Historical site tool API is running"}


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def safe_get(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def pick_first_attr(attributes: Dict[str, Any], keys: List[str]) -> str:
    lower_map = {str(k).lower(): v for k, v in safe_dict(attributes).items()}
    for key in keys:
        value = lower_map.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def is_likely_nsw(lat: float, lng: float) -> bool:
    return (-37.8 <= lat <= -28.0) and (140.5 <= lng <= 154.5)


def query_arcgis_point_layer(layer_url: str, lat: float, lng: float) -> Optional[Dict[str, Any]]:
    params = {
        "f": "json",
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "false",
        "geometry": f"{lng},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "resultRecordCount": 1,
    }
    data = safe_get(f"{layer_url}/query", params)
    if not data:
        return None
    features = data.get("features") or []
    if not features:
        return None
    return safe_dict(features[0].get("attributes"))


def build_surface_geology_context(resolved: Dict[str, Any]) -> Optional[Dict[str, str]]:
    try:
        lat = float(resolved.get("lat"))
        lng = float(resolved.get("lng"))
    except Exception:
        return None

    attributes = None
    source_name = ""
    source_note = ""

    if is_likely_qld(lat, lng):
        attributes = query_arcgis_point_layer(QLD_SURFACE_GEOLOGY_LAYER_URL, lat, lng)
        source_name = "QSpatial / GeologyDetailed - Detailed surface geology"
        source_note = "State of Queensland (Department of Resources), CC BY 4.0"
    elif is_likely_nsw(lat, lng):
        attributes = query_arcgis_point_layer(NSW_SEAMLESS_GEOLOGY_LAYER_URL, lat, lng)
        source_name = "NSW Seamless Geology dataset"
        source_note = "Geological Survey of New South Wales"

    if not attributes:
        return None

    unit_name = pick_first_attr(attributes, [
        "ru_name", "rockunit", "rock_unit", "unit_name", "name", "strat_name",
        "geolunit", "legend", "mapunit", "unitname", "formation", "unit"
    ])
    unit_code = pick_first_attr(attributes, [
        "ru_symbol", "symbol", "unit_code", "map_symbol", "code", "label", "unitcode", "mapunit_symbol"
    ])
    age = pick_first_attr(attributes, [
        "age", "age_name", "era", "period", "epoch", "max_age", "min_age", "age_text", "unit_age"
    ])
    lithology = pick_first_attr(attributes, [
        "lithology", "lith_desc", "lithological_description", "description", "desc_",
        "rocktype", "rock_type", "dominant_lithology", "lith", "unit_desc", "lith_sum", "lith_desc_"
    ])

    if not any([unit_name, unit_code, age, lithology]):
        readable_values = [
            str(v).strip()
            for k, v in attributes.items()
            if v is not None
            and str(v).strip()
            and str(v).strip().lower() not in ("null", "none")
            and not str(k).lower().endswith(("id", "fid", "objectid", "shape", "area", "len", "length"))
        ]
        if readable_values:
            unit_name = readable_values[0]
            if len(readable_values) > 1:
                lithology = readable_values[1]

    if not any([unit_name, unit_code, age, lithology]):
        return None

    return {
        "unit_name": unit_name,
        "unit_code": unit_code,
        "age": age,
        "lithology": lithology,
        "source_name": source_name,
        "source_note": source_note,
    }


def format_surface_geology_context(context: Optional[Dict[str, str]]) -> str:
    if not context:
        return (
            "Mapped surface geology could not be automatically retrieved for this site from the available public geology service."
            "<br/><br/>Mapped geology is provided for regional context only and does not replace intrusive geotechnical investigation."
            "<br/><br/>A detailed geotechnical investigation is required to confirm actual ground conditions and site classification in accordance with AS2870."
        )

    unit_name = safe_str(context.get("unit_name"), "").strip()
    unit_code = safe_str(context.get("unit_code"), "").strip()
    age = safe_str(context.get("age"), "").strip()
    lithology_raw = safe_str(context.get("lithology"), "").strip()
    source_name = safe_str(context.get("source_name"), "public geological mapping")
    source_note = safe_str(context.get("source_note"), "")

    age_upper = age.upper() if age else ""
    code_upper = unit_code.upper()
    name_upper = unit_name.upper()
    lith_upper = lithology_raw.upper()

    # QLD detailed geology can return a broad database field such as
    # "STRATIFIED UNIT...". For the PDF we convert common Quaternary coastal
    # units into a more useful geotechnical screening description.
    is_qc_quaternary = (
        "QC" in code_upper
        or "QC" in name_upper
        or "QUATERNARY" in age_upper
        or "QUATERNARY" in name_upper
    )

    if is_qc_quaternary and (
        not lithology_raw
        or "STRATIFIED UNIT" in lith_upper
        or "VOLCANIC" in lith_upper
        or "METAMORPHIC" in lith_upper
    ):
        geology_description = (
            "Qc / QLD Quaternary coastal plain deposits, typically comprising sand, muddy sand, "
            "minor mud and peat associated with undifferentiated swamp, tidal flat, beach-ridge and dune deposits."
        )
    elif lithology_raw:
        geology_description = lithology_raw
    else:
        geology_description = "mapped Quaternary surface deposits."

    unit_bits = []
    if unit_name:
        unit_bits.append(unit_name)
    if unit_code and unit_code.lower() not in unit_name.lower():
        unit_bits.append(f"({unit_code})")
    unit_text = " ".join(unit_bits) if unit_bits else "the mapped surface geology unit"

    first_sentence = f"The site is mapped within or near {unit_text}"
    if age:
        first_sentence += f", interpreted as {age}"
    first_sentence += "."

    source_sentence = f"<br/><br/>Source: {source_name}"
    if source_note:
        source_sentence += f" — {source_note}"
    source_sentence += ". Mapping is regional only and should not be relied on as confirmed ground conditions."

    limitation_sentence = (
        "<br/><br/>Mapped geology may not identify local fill, reclamation, weathering, demolition, service trenches, "
        "drainage changes or previous earthworks. A detailed geotechnical investigation is required to confirm "
        "actual subsurface conditions and AS2870 site classification."
    )

    return first_sentence + "<br/><br/>" + geology_description + source_sentence + limitation_sentence


def build_surface_geology_context_image(resolved: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        lat = float(resolved.get("lat"))
        lng = float(resolved.get("lng"))
    except Exception:
        return None

    buffer_deg = 0.012
    bbox = {
        "xmin": lng - buffer_deg,
        "ymin": lat - buffer_deg,
        "xmax": lng + buffer_deg,
        "ymax": lat + buffer_deg,
    }

    if is_likely_qld(lat, lng):
        url = (
            "https://spatial-gis.information.qld.gov.au/arcgis/rest/services/"
            "GeoscientificInformation/GeologyDetailed/MapServer/export"
            f"?bbox={bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
            "&bboxSR=4326"
            "&imageSR=4326"
            "&size=900,620"
            "&dpi=120"
            "&format=png32"
            "&transparent=true"
            "&f=image"
            "&layers=show:15"
        )
        source = "QSpatial / GeologyDetailed - Detailed surface geology"
    elif is_likely_nsw(lat, lng):
        url = (
            "https://gs-seamless.geoscience.nsw.gov.au/arcgis/rest/services/"
            "Geology/Geology_100K/MapServer/export"
            f"?bbox={bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
            "&bboxSR=4326"
            "&imageSR=4326"
            "&size=900,620"
            "&dpi=120"
            "&format=png32"
            "&transparent=true"
            "&f=image"
        )
        source = "NSW Seamless Geology dataset"
    else:
        return None

    return {
        "label": "surface_regional_geology_context",
        "type": "geology_context",
        "year": "current",
        "capture_date": None,
        "source": source,
        "url": url,
        "bbox": bbox,
    }


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_str(value: Any, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sanitize_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "site-history-report"


def epoch_ms_to_iso(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date().isoformat()
    except Exception:
        return None


def ensure_findings_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {
        "former_ponds_dams": {"status": "possible", "confidence": "low", "notes": ""},
        "vegetation_clearing": {"status": "none", "confidence": "low", "notes": ""},
        "fill_or_disturbance": {"status": "possible", "confidence": "low", "notes": ""},
    }


def ensure_closed_polygon(polygon: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
    if not polygon:
        return polygon
    if polygon[0] != polygon[-1]:
        return polygon + [polygon[0]]
    return polygon


def polygon_centroid(polygon: List[List[float]]) -> Dict[str, float]:
    lngs = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    return {"lng": sum(lngs) / len(lngs), "lat": sum(lats) / len(lats)}


def polygon_bbox(polygon: List[List[float]]) -> Dict[str, float]:
    lngs = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    return {
        "xmin": min(lngs),
        "ymin": min(lats),
        "xmax": max(lngs),
        "ymax": max(lats),
    }


def polygon_area_m2(polygon: Optional[List[List[float]]]) -> Optional[float]:
    polygon = ensure_closed_polygon(polygon)
    if not polygon or len(polygon) < 4:
        return None

    centroid = polygon_centroid(polygon)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(centroid["lat"]))
    if abs(meters_per_deg_lng) < 1e-6:
        meters_per_deg_lng = 1e-6

    pts = [((lng * meters_per_deg_lng), (lat * meters_per_deg_lat)) for lng, lat in polygon]
    area2 = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        area2 += (x1 * y2) - (x2 * y1)
    return abs(area2) / 2.0


def expand_bbox_meters(bbox: Dict[str, float], center_lat: float, pad_m: float = 8) -> Dict[str, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(center_lat))
    if abs(meters_per_deg_lng) < 1e-6:
        meters_per_deg_lng = 1e-6

    dlat = pad_m / meters_per_deg_lat
    dlng = pad_m / meters_per_deg_lng

    return {
        "xmin": bbox["xmin"] - dlng,
        "ymin": bbox["ymin"] - dlat,
        "xmax": bbox["xmax"] + dlng,
        "ymax": bbox["ymax"] + dlat,
    }


def bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    return ((bbox["xmin"] + bbox["xmax"]) / 2, (bbox["ymin"] + bbox["ymax"]) / 2)


def bbox_to_string(bbox: Dict[str, float]) -> str:
    return f'{bbox["xmin"]},{bbox["ymin"]},{bbox["xmax"]},{bbox["ymax"]}'


def bbox_width_height_m(bbox: Dict[str, float], center_lat: float) -> Dict[str, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(center_lat))
    width_m = (bbox["xmax"] - bbox["xmin"]) * meters_per_deg_lng
    height_m = (bbox["ymax"] - bbox["ymin"]) * meters_per_deg_lat
    return {"width_m": abs(width_m), "height_m": abs(height_m)}


def make_square_bbox_from_point(lat: float, lng: float, chip_size_m: float = 180) -> Dict[str, float]:
    half_size_m = chip_size_m / 2
    meters_per_deg_lat = 111320.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(lat))
    if abs(meters_per_deg_lng) < 1e-6:
        meters_per_deg_lng = 1e-6

    dlat = half_size_m / meters_per_deg_lat
    dlng = half_size_m / meters_per_deg_lng

    return {
        "xmin": lng - dlng,
        "ymin": lat - dlat,
        "xmax": lng + dlng,
        "ymax": lat + dlat,
    }


def split_bbox_into_subzones(bbox: Dict[str, float], center_lat: float) -> List[Dict[str, Any]]:
    cx, cy = bbox_center(bbox)
    zones = [
        {"label": "subzone_nw", "bbox": {"xmin": bbox["xmin"], "ymin": cy, "xmax": cx, "ymax": bbox["ymax"]}},
        {"label": "subzone_ne", "bbox": {"xmin": cx, "ymin": cy, "xmax": bbox["xmax"], "ymax": bbox["ymax"]}},
        {"label": "subzone_sw", "bbox": {"xmin": bbox["xmin"], "ymin": bbox["ymin"], "xmax": cx, "ymax": cy}},
        {"label": "subzone_se", "bbox": {"xmin": cx, "ymin": bbox["ymin"], "xmax": bbox["xmax"], "ymax": cy}},
        {"label": "subzone_west", "bbox": {"xmin": bbox["xmin"], "ymin": bbox["ymin"], "xmax": cx, "ymax": bbox["ymax"]}},
        {"label": "subzone_east", "bbox": {"xmin": cx, "ymin": bbox["ymin"], "xmax": bbox["xmax"], "ymax": bbox["ymax"]}},
    ]
    padded = []
    for zone in zones:
        padded.append({
            "label": zone["label"],
            "bbox": expand_bbox_meters(zone["bbox"], center_lat=center_lat, pad_m=4),
        })
    return padded




def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n - 1):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            xinters = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside




def polygon_membership_score(geo_bbox: Dict[str, float], polygon: List[List[float]]) -> int:
    if not geo_bbox or not polygon:
        return 0
    pts = [
        (geo_bbox["xmin"], geo_bbox["ymin"]),
        (geo_bbox["xmin"], geo_bbox["ymax"]),
        (geo_bbox["xmax"], geo_bbox["ymin"]),
        (geo_bbox["xmax"], geo_bbox["ymax"]),
        geo_bbox_centroid(geo_bbox),
    ]
    return sum(1 for pt in pts if point_in_polygon(pt, polygon))


def resolve_feature_relation(
    geo_bbox: Dict[str, float],
    polygon: Optional[List[List[float]]],
    feature_type: str = "",
    notes: str = "",
) -> str:
    if not geo_bbox or not polygon:
        return "on_site"

    cx, cy = geo_bbox_centroid(geo_bbox)
    if point_in_polygon((cx, cy), polygon):
        return "on_site"

    membership = polygon_membership_score(geo_bbox, polygon)
    feature_type = safe_str(feature_type, "")
    notes = safe_str(notes, "").lower()

    if membership >= 2:
        return "on_site"

    if is_water_feature_type(feature_type) or feature_type == "drainage_feature":
        if membership >= 1:
            return "on_site"
        if any(k in notes for k in ["former pond", "altered wet depression", "historical water", "secondary pond", "buried pond"]):
            return "on_site"

    return "adjacent"


def compute_feature_persistence_score(feature: Dict[str, Any]) -> int:
    years = merge_years(feature.get("detected_in_years", []))
    labels = merge_string_lists(
        [safe_str(feature.get("primary_image_label"), ""), safe_str(feature.get("detected_on_image"), "")]
        + safe_list(feature.get("source_image_labels"))
    )
    score = 0
    score += min(4, len(years))
    score += min(3, len(set(labels)))
    if any(lbl.startswith("historical_") for lbl in labels):
        score += 2
    if any(lbl.startswith("current_") for lbl in labels):
        score += 1
    if any("subzone" in lbl for lbl in labels):
        score += 1
    score += confidence_rank(safe_str(feature.get("confidence"), "low")) - 1
    return max(0, score)


def clamp_norm_bbox(bbox: List[float], min_size: float = 0.025, max_size: float = 0.22) -> List[float]:
    if len(bbox) != 4:
        return []
    try:
        x, y, w, h = [float(v) for v in bbox]
    except Exception:
        return []
    w = max(min_size, min(max_size, w))
    h = max(min_size, min(max_size, h))
    x = max(0.0, min(1.0 - w, x))
    y = max(0.0, min(1.0 - h, y))
    return [x, y, w, h]


def bbox_centroid_norm(bbox: List[float]) -> Optional[Tuple[float, float]]:
    if len(bbox) != 4:
        return None
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def norm_bbox_to_geo_bbox(norm_bbox: List[float], image_bbox: Dict[str, float]) -> Optional[Dict[str, float]]:
    if len(norm_bbox) != 4 or not image_bbox:
        return None
    x, y, w, h = norm_bbox
    xmin = image_bbox["xmin"] + x * (image_bbox["xmax"] - image_bbox["xmin"])
    xmax = image_bbox["xmin"] + (x + w) * (image_bbox["xmax"] - image_bbox["xmin"])
    ymax = image_bbox["ymax"] - y * (image_bbox["ymax"] - image_bbox["ymin"])
    ymin = image_bbox["ymax"] - (y + h) * (image_bbox["ymax"] - image_bbox["ymin"])
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def geo_bbox_to_norm_bbox(geo_bbox: Dict[str, float], image_bbox: Dict[str, float]) -> Optional[List[float]]:
    if not geo_bbox or not image_bbox:
        return None
    width = image_bbox["xmax"] - image_bbox["xmin"]
    height = image_bbox["ymax"] - image_bbox["ymin"]
    if abs(width) < 1e-12 or abs(height) < 1e-12:
        return None

    x = (geo_bbox["xmin"] - image_bbox["xmin"]) / width
    w = (geo_bbox["xmax"] - geo_bbox["xmin"]) / width
    y = (image_bbox["ymax"] - geo_bbox["ymax"]) / height
    h = (geo_bbox["ymax"] - geo_bbox["ymin"]) / height

    clipped_x1 = max(0.0, min(1.0, x))
    clipped_y1 = max(0.0, min(1.0, y))
    clipped_x2 = max(0.0, min(1.0, x + w))
    clipped_y2 = max(0.0, min(1.0, y + h))

    w2 = clipped_x2 - clipped_x1
    h2 = clipped_y2 - clipped_y1
    if w2 <= 0.01 or h2 <= 0.01:
        return None
    return [clipped_x1, clipped_y1, w2, h2]


def geo_bbox_centroid(geo_bbox: Dict[str, float]) -> Tuple[float, float]:
    return (
        (geo_bbox["xmin"] + geo_bbox["xmax"]) / 2.0,
        (geo_bbox["ymin"] + geo_bbox["ymax"]) / 2.0,
    )


def feature_sort_key_for_ids(feature: Dict[str, Any]) -> Tuple[float, float]:
    geo_bbox = safe_dict(feature.get("geo_bbox"))
    if geo_bbox:
        cx, cy = geo_bbox_centroid(geo_bbox)
        return (-cy, cx)
    return (999.0, 999.0)


def overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    ixmin = max(a["xmin"], b["xmin"])
    iymin = max(a["ymin"], b["ymin"])
    ixmax = min(a["xmax"], b["xmax"])
    iymax = min(a["ymax"], b["ymax"])
    if ixmax <= ixmin or iymax <= iymin:
        return 0.0
    inter = (ixmax - ixmin) * (iymax - iymin)
    area_a = max(1e-12, (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"]))
    area_b = max(1e-12, (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"]))
    return inter / min(area_a, area_b)


def geo_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 999.0
    ax, ay = geo_bbox_centroid(a)
    bx, by = geo_bbox_centroid(b)
    return math.hypot(ax - bx, ay - by)



def geo_bbox_area(b: Dict[str, float]) -> float:
    if not b:
        return 0.0
    return max(1e-12, abs((b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])))


def geo_bbox_size_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    area_a = geo_bbox_area(a)
    area_b = geo_bbox_area(b)
    return min(area_a, area_b) / max(area_a, area_b) if area_a > 0 and area_b > 0 else 0.0


def feature_year_set(feature: Dict[str, Any]) -> set:
    return {
        int(y) for y in safe_list(feature.get("detected_in_years"))
        if isinstance(y, (int, float))
    }


def temporal_compatibility_score(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    y1 = feature_year_set(f1)
    y2 = feature_year_set(f2)
    if not y1 or not y2:
        return 0.0
    inter = len(y1 & y2)
    union = len(y1 | y2)
    return inter / union if union else 0.0

def should_merge_features(f1: Dict[str, Any], f2: Dict[str, Any]) -> bool:
    b1 = safe_dict(f1.get("geo_bbox"))
    b2 = safe_dict(f2.get("geo_bbox"))
    if not b1 or not b2:
        return False
    t1 = safe_str(f1.get("feature_type"), "other")
    t2 = safe_str(f2.get("feature_type"), "other")
    r1 = safe_str(f1.get("location_relation"), "")
    r2 = safe_str(f2.get("location_relation"), "")
    if r1 and r2 and r1 != r2:
        return False

    overlap = overlap_ratio(b1, b2)
    distance = geo_distance(b1, b2)
    size_ratio = geo_bbox_size_ratio(b1, b2)
    same_water_family = is_water_feature_type(t1) and is_water_feature_type(t2)

    if same_water_family:
        # Allow same pond detections across site/context scales to merge,
        # but never merge nearby separate ponds on distance alone.
        if distance > 0.00015:
            return False
        if overlap >= 0.55 and size_ratio >= 0.50:
            return True
        if overlap >= 0.30 and distance <= 0.00006 and size_ratio >= 0.55:
            return True
        return False

    if t1 == t2:
        if overlap > 0.60:
            return True
        if t1 == "disturbance" and distance < 0.00010:
            return True
    return False

def feature_note_tokens(text: str) -> set:
    t = re.sub(r"[^a-z0-9 ]+", " ", safe_str(text).lower())
    stop = {
        "the","a","an","and","or","of","with","in","on","to","from","since","through",
        "visible","imagery","image","site","water","body","feature","pond","former","candidate"
    }
    return {tok for tok in t.split() if len(tok) >= 4 and tok not in stop}


def feature_note_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ta = feature_note_tokens(safe_str(a.get("notes"), ""))
    tb = feature_note_tokens(safe_str(b.get("notes"), ""))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def is_secondary_pond_like(feature: Dict[str, Any]) -> bool:
    text = (safe_str(feature.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(feature.get("evidence"))])).lower()
    return any(k in text for k in ["smaller", "secondary", "south of main", "south of the main", "less visible", "dry in recent", "subtle circular"])


def collapse_near_identical_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final: List[Dict[str, Any]] = []
    for feature in features:
        matched = False
        for existing in final:
            t1 = safe_str(existing.get("feature_type"), "other")
            t2 = safe_str(feature.get("feature_type"), "other")
            if not (is_water_feature_type(t1) and is_water_feature_type(t2)):
                continue
            if cluster_family(existing) != cluster_family(feature):
                continue
            if safe_str(existing.get("location_relation"), "") != safe_str(feature.get("location_relation"), ""):
                continue
            if is_secondary_pond_like(existing) != is_secondary_pond_like(feature):
                continue

            b1 = safe_dict(existing.get("geo_bbox"))
            b2 = safe_dict(feature.get("geo_bbox"))
            if not b1 or not b2:
                continue

            dist = geo_distance(b1, b2)
            overlap = overlap_ratio(b1, b2)
            size_ratio = geo_bbox_size_ratio(b1, b2)
            note_sim = feature_note_similarity(existing, feature)
            year_overlap = temporal_compatibility_score(existing, feature) > 0.0

            should_merge = False
            if overlap >= 0.55 and size_ratio >= 0.50:
                should_merge = True
            elif (
                dist < 0.00005
                and size_ratio >= 0.75
                and year_overlap
                and note_sim >= 0.08
                and overlap >= 0.12
            ):
                should_merge = True

            if should_merge:
                merge_feature_pair(existing, feature)
                matched = True
                break
        if not matched:
            final.append(dict(feature))
    return final

def dedupe_final_anchored_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for feature in features:
        matched = False
        for existing in cleaned:
            if cluster_family(existing) != cluster_family(feature):
                continue
            if feature_relation_bucket(existing) != feature_relation_bucket(feature):
                continue
            b1 = safe_dict(existing.get("geo_bbox"))
            b2 = safe_dict(feature.get("geo_bbox"))
            if not b1 or not b2:
                continue
            overlap = overlap_ratio(b1, b2)
            dist = geo_distance(b1, b2)
            size_ratio = geo_bbox_size_ratio(b1, b2)
            if cluster_family(feature) == "water":
                if overlap >= 0.55 and size_ratio >= 0.50:
                    merge_feature_pair(existing, feature)
                    matched = True
                    break
                if overlap >= 0.30 and dist <= 0.00006 and size_ratio >= 0.55:
                    merge_feature_pair(existing, feature)
                    matched = True
                    break
            elif overlap >= 0.70:
                merge_feature_pair(existing, feature)
                matched = True
                break
        if not matched:
            cleaned.append(dict(feature))
    return cleaned

def likely_current_visible(feature: Dict[str, Any]) -> bool:
    notes = safe_str(feature.get("notes"), "").lower()
    evidence_text = " ".join([str(x) for x in safe_list(feature.get("evidence"))]).lower()
    years = [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
    if "no open water" in notes or "infilled" in notes or "drained" in notes or "former" in notes:
        return False
    if "still visible" in notes or "current imagery" in evidence_text or "currently present" in notes:
        return True
    return any(y >= 2020 for y in years)


def choose_canonical_current_label(feature: Dict[str, Any], image_lookup: Dict[str, Dict[str, Any]]) -> str:
    preferred = ["current_wide_context", "current_context", "current_site"]
    geo_bbox = safe_dict(feature.get("geo_bbox"))
    for label in preferred:
        img = image_lookup.get(label)
        if not img:
            continue
        img_bbox = safe_dict(img.get("bbox"))
        if geo_bbox_to_norm_bbox(geo_bbox, img_bbox):
            return label
    return safe_str(feature.get("primary_image_label"), "current_wide_context")


def is_current_label(label: str) -> bool:
    return safe_str(label).startswith("current_")


def is_historical_label(label: str) -> bool:
    return safe_str(label).startswith("historical_")


def water_signal_score(text: str) -> int:
    t = text.lower()
    score = 0
    for key in [
        "water", "open water", "waterbody", "water body", "pond", "wet",
        "standing water", "basin", "depression", "vegetated ring", "circular",
        "round", "former pond", "infilled pond"
    ]:
        if key in t:
            score += 1
    return score


def disturbance_signal_score(text: str) -> int:
    t = text.lower()
    score = 0
    for key in [
        "fill", "disturb", "earthworks", "bare soil", "cleared soil", "material pile",
        "construction", "access road", "structure", "debris", "graded", "compacted",
        "stockpile", "retaining", "cut platform", "benched", "benching", "dwelling",
        "shed", "slab", "pad", "excavation", "batter", "machine tracks", "dozer tracks",
        "track marks", "tracked ground", "bulk earthworks", "platform preparation",
        "earthmoving", "scraped ground", "stripped ground", "cut batter", "fill batter",
        "bench", "terraced", "terracing", "reworked ground", "site preparation"
    ]:
        if key in t:
            score += 1
    return score


def building_signal_score(text: str) -> int:
    t = text.lower()
    score = 0
    for key in [
        "building", "dwelling", "house", "shed", "slab", "hardstand", "driveway",
        "roof", "structure", "building footprint", "house pad"
    ]:
        if key in t:
            score += 1
    return score


def disturbance_or_building_blob(feature: Dict[str, Any]) -> str:
    return (
        safe_str(feature.get("notes"), "") + " " +
        " ".join([str(x) for x in safe_list(feature.get("evidence"))])
    ).lower()


def has_structure_or_hardstand_signal(text: str) -> bool:
    return building_signal_score(text) >= 1


def has_strong_disturbance_signal(text: str) -> bool:
    return disturbance_signal_score(text) >= 1


def merge_years(values: List[Any]) -> List[int]:
    return sorted(set([int(y) for y in safe_list(values) if isinstance(y, (int, float))]))


def merge_string_lists(values: List[Any]) -> List[str]:
    out = []
    seen = set()
    for item in safe_list(values):
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def choose_best_annotation_label(feature: Dict[str, Any], image_lookup: Dict[str, Dict[str, Any]]) -> str:
    detected = safe_str(feature.get("detected_on_image"), "")
    if detected and detected in image_lookup:
        return detected

    primary = safe_str(feature.get("primary_image_label"), "")
    if primary and primary in image_lookup:
        return primary

    labels = merge_string_lists(feature.get("source_image_labels", []))
    for lbl in labels:
        if lbl in image_lookup:
            return lbl

    return "current_wide_context"

def classify_cluster_feature(feature: Dict[str, Any], image_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    f = dict(feature)

    notes = safe_str(f.get("notes"), "")
    evidence = merge_string_lists(f.get("evidence", []))
    text = (notes + " " + " ".join(evidence)).lower()

    labels = merge_string_lists(f.get("source_image_labels", []))
    years = merge_years(f.get("detected_in_years", []))

    current_labels = [lbl for lbl in labels if is_current_label(lbl)]
    historical_labels = [lbl for lbl in labels if is_historical_label(lbl)]
    historical_seen = bool(historical_labels) or any(y <= 2015 for y in years)
    current_seen = bool(current_labels) or any(y >= 2020 for y in years)

    water_score = water_signal_score(text)
    disturbance_score = disturbance_signal_score(text)
    building_score = building_signal_score(text)

    raw_type = safe_str(f.get("feature_type"), "other").lower()

    is_canal_signal = any(k in text for k in [
        "canal", "estate canal", "tidal canal", "linear canal", "rear boundary canal",
        "waterway", "revetment", "pontoon", "jetty", "boat ramp"
    ])
    is_linear_drainage_signal = any(k in text for k in [
        "creek", "drainage", "drain", "linear water", "channel", "flow path"
    ])
    is_beach_signal = any(k in text for k in [
        "beach", "foreshore", "dune", "tidal", "estuary", "ocean", "shoreline", "coastal", "bay"
    ])
    is_reclaimed_signal = any(k in text for k in [
        "reclaimed", "reclamation", "dredged", "canal fill", "former canal", "filled canal", "marsh", "low-lying"
    ])

    if raw_type in ("existing_structure", "former_structure", "hardstand_or_slab"):
        final_type = raw_type
    elif raw_type in ("beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature", "possible_reclaimed_ground", "retaining_or_cut_fill", "significant_tree_or_vegetation"):
        final_type = raw_type
    elif raw_type == "canal" or is_canal_signal:
        final_type = "canal"
    elif raw_type == "drainage_feature" or (is_linear_drainage_signal and not is_canal_signal):
        final_type = "drainage_feature"
    elif is_beach_signal:
        final_type = "beach_foreshore_or_coastal_edge"
    elif is_reclaimed_signal and not any(k in text for k in ["isolated pond", "pond footprint", "rounded basin", "circular"]):
        final_type = "possible_reclaimed_ground"
    else:
        ever_looked_like_pond = (
            raw_type in ("water_candidate", "probable_pond", "pond", "former_pond")
            or water_score >= 1
            or any(k in text for k in [
                "pond", "water", "wet", "basin", "depression",
                "vegetated ring", "circular", "round", "former pond",
                "wet area", "wet hollow", "water body", "waterbody"
            ])
        )

        strong_current_open_water = (
            current_seen and (
                any(k in text for k in [
                    "current water", "current open water", "open water currently",
                    "standing water visible", "visible water currently present",
                    "still visible in current imagery", "current water visible",
                    "visible in current imagery with water", "open water"
                ])
                or ("dark water" in text and "current" in text)
            )
        )

        strong_building_only = building_score >= 1 and water_score == 0
        mixed_but_disturbance_led = (building_score >= 1 or disturbance_score >= 2) and not strong_current_open_water and water_score <= 1

        if strong_building_only:
            final_type = "existing_structure" if current_seen else "former_structure" if historical_seen else "existing_structure"
        elif mixed_but_disturbance_led and not historical_seen:
            final_type = "disturbance"
        elif disturbance_score >= 3 and not ever_looked_like_pond:
            final_type = "disturbance"
        elif ever_looked_like_pond and disturbance_score < 2 and building_score == 0:
            if strong_current_open_water or (current_seen and likely_current_visible(f)):
                final_type = "pond"
            elif historical_seen:
                final_type = "former_pond"
            elif current_seen:
                final_type = "pond"
            else:
                final_type = "probable_pond"
        elif ever_looked_like_pond and historical_seen and water_score >= 2 and building_score == 0:
            final_type = "former_pond" if not current_seen else "pond"
        elif disturbance_score >= 1 or building_score >= 1:
            final_type = "disturbance"
        else:
            final_type = "other"

    f["feature_type"] = final_type
    f["source_image_labels"] = labels
    f["detected_in_years"] = years
    f["evidence"] = evidence
    f["persistence_score"] = compute_feature_persistence_score(f)

    if final_type == "former_pond":
        f["risk_priority"] = "primary" if safe_str(f.get("location_relation"), "") == "on_site" else "contextual"
    elif final_type in ("pond", "probable_pond", "disturbance", "existing_structure", "former_structure", "hardstand_or_slab", "possible_reclaimed_ground", "retaining_or_cut_fill", "significant_tree_or_vegetation"):
        f["risk_priority"] = "secondary" if safe_str(f.get("location_relation"), "") == "on_site" else "contextual"
    elif final_type in ("drainage_feature", "canal", "beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"):
        f["risk_priority"] = "contextual" if safe_str(f.get("location_relation"), "") != "on_site" else "secondary"
    else:
        f["risk_priority"] = safe_str(f.get("risk_priority"), "secondary")

    if not safe_str(f.get("detected_on_image"), ""):
        f["detected_on_image"] = safe_str(f.get("primary_image_label"), "")

    if not safe_str(f.get("primary_image_label"), ""):
        f["primary_image_label"] = choose_best_annotation_label(f, image_lookup)

    return f

def dedupe_preserve_order(items: List[Any]) -> List[Any]:
    out = []
    seen = set()
    for item in safe_list(items):
        key = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def dedupe_sentences(text_value: str) -> str:
    text_value = safe_str(text_value, "").strip()
    if not text_value:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text_value) if p.strip()]
    out = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return " ".join(out)


def dedupe_limitations(items: List[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in safe_list(items):
        s = str(item).strip()
        if not s or s.startswith("{") or '"summary"' in s:
            continue
        key = re.sub(r"[^a-z0-9]", "", s.lower())[:80]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out or ["Imagery interpretation remains preliminary and should be manually reviewed where uncertainty persists."]


def image_scale_rank(label: str) -> int:
    label = safe_str(label, "")
    if "subzone" in label:
        return 4
    if label.startswith("current_site") or label.startswith("historical_qld_followup_") or label.startswith("historical_qld_"):
        return 3
    if "context" in label:
        return 2
    if "wide_context" in label:
        return 1
    return 0


def canonical_image_rank(label: str) -> int:
    label = safe_str(label, "")
    # Stable anchors should prefer full-frame historical/site images over subzones,
    # because subzones can drift the same feature onto a slightly different footprint.
    if label.startswith("historical_qld_followup_") or label.startswith("historical_qld_"):
        return 5 if "subzone" not in label and "context" not in label else 2
    if label.startswith("current_site"):
        return 4
    if "context" in label and "subzone" not in label:
        return 3
    if "wide_context" in label:
        return 2
    if "subzone" in label:
        return 1
    return 0


def feature_relation_bucket(feature: Dict[str, Any]) -> str:
    relation = safe_str(feature.get("location_relation"), "on_site")
    return "on_site" if relation == "on_site" else "adjacent"


def feature_text_blob(feature: Dict[str, Any]) -> str:
    return (
        safe_str(feature.get("notes"), "") + " " +
        " ".join([str(x) for x in safe_list(feature.get("evidence"))])
    ).lower()


def feature_current_visible(feature: Dict[str, Any]) -> bool:
    text = feature_text_blob(feature)
    years = [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
    labels = merge_string_lists(
        [safe_str(feature.get("primary_image_label"), ""), safe_str(feature.get("detected_on_image"), "")] +
        safe_list(feature.get("source_image_labels"))
    )
    if any(k in text for k in [
        "absent in current", "not visible in recent", "no clear water in 2021",
        "dry in recent", "disappeared in recent", "infilled", "former pond", "no longer visible"
    ]):
        return False
    if any(k in text for k in [
        "current water", "current open water", "visible water currently", "still visible in current",
        "current imagery with water", "open water currently present", "wet soils seen"
    ]):
        return True
    if any(lbl.startswith("current_") for lbl in labels) and safe_str(feature.get("feature_type"), "") in ("pond", "probable_pond"):
        return True
    return any(y >= 2020 for y in years) and safe_str(feature.get("feature_type"), "") == "pond"


def feature_historical_visible(feature: Dict[str, Any]) -> bool:
    text = feature_text_blob(feature)
    years = [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
    if any(y <= 2019 for y in years):
        return True
    return any(k in text for k in ["historical water", "visible in 1995", "older imagery", "historic imagery", "historically"])


def cluster_family(feature: Dict[str, Any]) -> str:
    ftype = safe_str(feature.get("feature_type"), "other")
    if ftype in ("pond", "former_pond", "probable_pond", "depression", "former_pond_or_dam", "pond_on_site"):
        return "water"
    if ftype in ("canal", "canal_edge_or_reclaimed_waterway", "possible_reclaimed_ground"):
        return "canal"
    if ftype in ("drainage_feature", "creek_or_drainage_line"):
        return "drainage"
    if ftype in ("beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"):
        return "water_context"
    if ftype in ("disturbance", "fill_area", "retaining_or_cut_fill"):
        return "disturbance"
    if ftype in ("existing_structure", "former_structure", "hardstand_or_slab"):
        return "structure"
    if ftype == "significant_tree_or_vegetation":
        return "vegetation"
    return "other"

def should_cluster_same_feature(f1: Dict[str, Any], f2: Dict[str, Any]) -> bool:
    b1 = safe_dict(f1.get("geo_bbox"))
    b2 = safe_dict(f2.get("geo_bbox"))
    if not b1 or not b2:
        return False

    fam1 = cluster_family(f1)
    fam2 = cluster_family(f2)
    if fam1 != fam2:
        return False
    if feature_relation_bucket(f1) != feature_relation_bucket(f2):
        return False

    dist = geo_distance(b1, b2)
    overlap = overlap_ratio(b1, b2)
    sim = feature_note_similarity(f1, f2)
    size_ratio = geo_bbox_size_ratio(b1, b2)
    temporal_score = temporal_compatibility_score(f1, f2)

    labels = merge_string_lists(
        [safe_str(f1.get("primary_image_label"), ""), safe_str(f1.get("detected_on_image"), "")] +
        safe_list(f1.get("source_image_labels")) +
        [safe_str(f2.get("primary_image_label"), ""), safe_str(f2.get("detected_on_image"), "")] +
        safe_list(f2.get("source_image_labels"))
    )
    max_scale = max([image_scale_rank(lbl) for lbl in labels] + [0])

    if fam1 == "water":
        # Allow same pond across site/context scales, but never cluster water on proximity alone.
        distance_threshold = 0.00020 if max_scale >= 4 else 0.00015
        if overlap >= 0.18 and size_ratio >= 0.45:
            return True
        if (
            dist <= distance_threshold
            and sim >= 0.08
            and size_ratio >= 0.45
            and (temporal_score > 0.0 or overlap >= 0.15)
        ):
            return True
        return False

    if fam1 == "disturbance":
        if overlap >= 0.30:
            return True
        if dist <= 0.00020 and sim >= 0.03:
            return True
        return False

    if fam1 == "drainage":
        return overlap >= 0.20 or dist <= 0.00020

    return overlap >= 0.40

def cluster_anchor_feature(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    def anchor_key(f: Dict[str, Any]):
        label = safe_str(f.get("detected_on_image"), "") or safe_str(f.get("primary_image_label"), "")
        years = [int(y) for y in safe_list(f.get("detected_in_years")) if isinstance(y, (int, float))]
        return (
            canonical_image_rank(label),
            int(f.get("persistence_score", 0) or 0),
            confidence_rank(safe_str(f.get("confidence"), "low")),
            len(years),
            feature_type_rank(safe_str(f.get("feature_type"), "other")),
            1 if feature_current_visible(f) else 0,
            len(safe_str(f.get("notes"), "")),
        )
    return sorted(features, key=anchor_key, reverse=True)[0]


def merge_cluster_to_feature(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    anchor = dict(cluster_anchor_feature(cluster))
    family = cluster_family(anchor)

    years = sorted(set(
        int(y)
        for f in cluster
        for y in safe_list(f.get("detected_in_years"))
        if isinstance(y, (int, float))
    ))
    source_labels = merge_string_lists(
        [safe_str(anchor.get("primary_image_label"), ""), safe_str(anchor.get("detected_on_image"), "")] +
        [lbl for f in cluster for lbl in safe_list(f.get("source_image_labels"))]
    )
    evidence = merge_string_lists([e for f in cluster for e in safe_list(f.get("evidence"))])

    relation = "on_site" if any(feature_relation_bucket(f) == "on_site" for f in cluster) else "adjacent"
    current_seen = any(feature_current_visible(f) for f in cluster)
    historical_seen = any(feature_historical_visible(f) for f in cluster)

    if family == "water":
        final_type = "pond" if current_seen else "former_pond"
        if not current_seen and historical_seen:
            final_type = "former_pond"
    elif family == "canal":
        # Keep canal / reclaimed waterway context separate from pond logic.
        final_type = "canal"
    elif family == "drainage":
        final_type = "drainage_feature"
    elif family == "water_context":
        final_type = safe_str(anchor.get("feature_type"), "uncertain_water_related_feature")
    elif family == "disturbance":
        final_type = "disturbance"
    elif family == "structure":
        if any(safe_str(f.get("feature_type"), "") == "former_structure" for f in cluster):
            final_type = "former_structure"
        elif any(safe_str(f.get("feature_type"), "") == "hardstand_or_slab" for f in cluster):
            final_type = "hardstand_or_slab"
        else:
            final_type = safe_str(anchor.get("feature_type"), "existing_structure")
    elif family == "vegetation":
        final_type = "significant_tree_or_vegetation"
    else:
        final_type = safe_str(anchor.get("feature_type"), "other")

    out = dict(anchor)
    out["feature_type"] = final_type
    out["bbox_locked"] = True
    out["location_relation"] = relation
    out["detected_in_years"] = years
    out["source_image_labels"] = source_labels
    out["evidence"] = evidence
    out["confidence"] = max(
        [safe_str(f.get("confidence"), "low") for f in cluster],
        key=lambda c: confidence_rank(c)
    )

    if family == "water" and current_seen:
        current_members = [f for f in cluster if feature_current_visible(f)]
        if current_members:
            best_current = cluster_anchor_feature(current_members)
            out["primary_image_label"] = safe_str(best_current.get("primary_image_label"), out.get("primary_image_label", ""))
            out["detected_on_image"] = safe_str(best_current.get("detected_on_image"), out.get("detected_on_image", ""))
            out["approximate_bbox_norm"] = safe_list(best_current.get("approximate_bbox_norm")) or safe_list(out.get("approximate_bbox_norm"))
            out["geo_bbox"] = safe_dict(best_current.get("geo_bbox")) or safe_dict(out.get("geo_bbox"))
    else:
        out["primary_image_label"] = safe_str(anchor.get("primary_image_label"), "")
        out["detected_on_image"] = safe_str(anchor.get("detected_on_image"), safe_str(anchor.get("primary_image_label"), ""))
        out["approximate_bbox_norm"] = safe_list(anchor.get("approximate_bbox_norm"))
        out["geo_bbox"] = safe_dict(anchor.get("geo_bbox"))

    notes = [safe_str(f.get("notes"), "") for f in cluster if safe_str(f.get("notes"), "")]
    out["notes"] = max(notes, key=len) if notes else safe_str(out.get("notes"), "")
    out["notes"] = dedupe_sentences(out["notes"])

    if relation != "on_site":
        out["risk_priority"] = "contextual"
    elif final_type == "former_pond":
        out["risk_priority"] = "primary"
    elif final_type in ("pond", "probable_pond", "disturbance", "fill_area", "existing_structure", "former_structure", "hardstand_or_slab", "retaining_or_cut_fill"):
        out["risk_priority"] = "secondary"
    elif final_type in ("canal", "drainage_feature", "beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"):
        out["risk_priority"] = "contextual"
    else:
        out["risk_priority"] = safe_str(out.get("risk_priority"), "secondary")

    out["feature_id"] = safe_str(anchor.get("feature_id"), "")
    return out

def cluster_features_by_identity(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remaining = [dict(f) for f in features]
    clusters: List[List[Dict[str, Any]]] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            still = []
            for candidate in remaining:
                if any(should_cluster_same_feature(member, candidate) for member in cluster):
                    cluster.append(candidate)
                    changed = True
                else:
                    still.append(candidate)
            remaining = still
        clusters.append(cluster)

    merged = [merge_cluster_to_feature(cluster) for cluster in clusters]
    return sorted(
        merged,
        key=lambda f: (
            0 if feature_relation_bucket(f) == "on_site" and cluster_family(f) == "water" else
            1 if feature_relation_bucket(f) == "on_site" else 2,
            *feature_sort_key_for_ids(f)
        )
    )

def is_water_feature_type(feature_type: str) -> bool:
    return feature_type in (
        "pond", "former_pond", "probable_pond", "depression",
        "pond_on_site", "former_pond_or_dam",
        "canal", "canal_edge_or_reclaimed_waterway",
        "drainage_feature", "creek_or_drainage_line",
        "beach_foreshore_or_coastal_edge", "large_external_waterbody",
        "uncertain_water_related_feature",
    )

def feature_type_rank(feature_type: str) -> int:
    ranks = {
        "former_pond": 8,
        "former_pond_or_dam": 8,
        "pond": 7,
        "pond_on_site": 7,
        "probable_pond": 6,
        "depression": 5,
        "canal": 5,
        "canal_edge_or_reclaimed_waterway": 5,
        "possible_reclaimed_ground": 5,
        "drainage_feature": 4,
        "creek_or_drainage_line": 4,
        "beach_foreshore_or_coastal_edge": 4,
        "large_external_waterbody": 3,
        "uncertain_water_related_feature": 3,
        "fill_area": 3,
        "disturbance": 3,
        "retaining_or_cut_fill": 3,
        "existing_structure": 2,
        "former_structure": 2,
        "hardstand_or_slab": 2,
        "significant_tree_or_vegetation": 1,
    }
    return ranks.get(feature_type, 0)

def confidence_rank(confidence: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(confidence, 0)


def sanitize_ai_feature_id(raw_id: str, feature_type: str) -> str:
    rid = safe_str(raw_id).strip()
    if not rid:
        return ""
    lowered = rid.lower()
    if lowered.startswith(("pond ", "former pond", "active pond", "probable pond", "water feature", "feature ")):
        return ""
    if feature_type in ("fill_area",):
        return "Fill Area"
    if feature_type in ("disturbance",):
        return "Disturbance Area"
    if feature_type in ("canal",):
        return "Adjacent Canal"
    if feature_type in ("beach_foreshore_or_coastal_edge",):
        return "Foreshore / Coastal Edge"
    if feature_type in ("possible_reclaimed_ground",):
        return "Possible Reclaimed Ground"
    return rid


def merge_feature_pair(existing: Dict[str, Any], feature: Dict[str, Any]) -> Dict[str, Any]:
    etype = safe_str(existing.get("feature_type"), "other")
    ftype = safe_str(feature.get("feature_type"), "other")
    if feature_type_rank(ftype) > feature_type_rank(etype):
        existing["feature_type"] = ftype
    elif feature_type_rank(ftype) == feature_type_rank(etype):
        if confidence_rank(safe_str(feature.get("confidence"), "low")) > confidence_rank(safe_str(existing.get("confidence"), "low")):
            existing["feature_type"] = ftype

    if confidence_rank(safe_str(feature.get("confidence"), "low")) > confidence_rank(safe_str(existing.get("confidence"), "low")):
        existing["confidence"] = feature.get("confidence", existing.get("confidence"))

    if safe_str(feature.get("risk_priority"), "") == "primary":
        existing["risk_priority"] = "primary"
    elif existing.get("risk_priority") != "primary" and safe_str(feature.get("risk_priority"), "") == "secondary":
        existing["risk_priority"] = "secondary"

    existing["persistence_score"] = max(
        int(existing.get("persistence_score", 0) or 0),
        int(feature.get("persistence_score", 0) or 0),
    )

    existing["detected_in_years"] = sorted(set(
        [int(y) for y in safe_list(existing.get("detected_in_years")) if isinstance(y, (int, float))] +
        [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
    ))
    existing["evidence"] = list(dict.fromkeys(
        [str(x) for x in safe_list(existing.get("evidence")) + safe_list(feature.get("evidence"))]
    ))
    existing["source_image_labels"] = list(dict.fromkeys(
        [str(x) for x in safe_list(existing.get("source_image_labels")) + safe_list(feature.get("source_image_labels"))]
    ))
    if len(safe_str(feature.get("notes"), "")) > len(safe_str(existing.get("notes"), "")):
        existing["notes"] = feature.get("notes", "")
    if not safe_dict(existing.get("geo_bbox")) and safe_dict(feature.get("geo_bbox")):
        existing["geo_bbox"] = feature.get("geo_bbox")
    if safe_str(existing.get("location_relation"), "") != "on_site" and safe_str(feature.get("location_relation"), "") == "on_site":
        existing["location_relation"] = "on_site"

    existing_detected = safe_str(existing.get("detected_on_image"), "")
    feature_detected = safe_str(feature.get("detected_on_image"), "")
    if not existing_detected and feature_detected:
        existing["detected_on_image"] = feature_detected

    if feature_detected and feature_detected == safe_str(feature.get("primary_image_label"), ""):
        if confidence_rank(safe_str(feature.get("confidence"), "low")) > confidence_rank(safe_str(existing.get("confidence"), "low")):
            existing["detected_on_image"] = feature_detected
            existing["primary_image_label"] = feature_detected
            existing["approximate_bbox_norm"] = safe_list(feature.get("approximate_bbox_norm"))

    return existing

def merge_feature_sets_geometry_first(base_features: List[Dict[str, Any]], extra_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    def try_merge(feature: Dict[str, Any]) -> bool:
        feature_geo = safe_dict(feature.get("geo_bbox"))
        if not feature_geo:
            return False
        for existing in merged:
            existing_geo = safe_dict(existing.get("geo_bbox"))
            if not existing_geo:
                continue
            if should_merge_features(existing, feature):
                merge_feature_pair(existing, feature)
                existing["source_image_labels"] = merge_string_lists(
                    safe_list(existing.get("source_image_labels")) + safe_list(feature.get("source_image_labels"))
                )
                return True
        return False

    for feature in base_features + extra_features:
        if not try_merge(feature):
            merged.append(dict(feature))

    return merged

def normalize_change_timeline_items(items: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in safe_list(items):
        if isinstance(item, dict):
            period = safe_str(item.get("period"), "")
            obs = safe_str(item.get("observation"), "")
            if period or obs:
                out.append({"period": period or "Unknown period", "observation": obs})
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            if ":" in text:
                period, obs = text.split(":", 1)
                out.append({"period": period.strip(), "observation": obs.strip()})
            else:
                out.append({"period": "General", "observation": text})
    deduped: List[Dict[str, str]] = []
    seen = set()
    for item in out:
        key = (item["period"], item["observation"])
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def normalize_feature_timeline_items(items: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in safe_list(items):
        if isinstance(item, dict):
            fid = safe_str(item.get("feature_id"), "Feature")
            if isinstance(item.get("timeline"), list):
                for t in safe_list(item.get("timeline")):
                    if not isinstance(t, dict):
                        continue
                    year = t.get("year")
                    status = safe_str(t.get("status"), safe_str(t.get("type"), ""))
                    notes = safe_str(t.get("notes"), "")
                    period = str(year) if year is not None else "Unknown"
                    observation = " ".join([x for x in [status, notes] if x]).strip()
                    out.append({"feature_id": fid, "period": period, "observation": observation})
            else:
                period = safe_str(item.get("period"), "")
                obs = safe_str(item.get("observation"), "")
                if period or obs:
                    out.append({"feature_id": fid, "period": period or "Unknown", "observation": obs})
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            if "—" in text and ":" in text:
                left, obs = text.split(":", 1)
                fid, period = left.split("—", 1)
                out.append({"feature_id": fid.strip(), "period": period.strip(), "observation": obs.strip()})
            else:
                out.append({"feature_id": "Feature", "period": "General", "observation": text})
    deduped: List[Dict[str, str]] = []
    seen = set()
    for item in out:
        key = (item["feature_id"], item["period"], item["observation"])
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def final_consolidate_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not features:
        return []

    merged: List[Dict[str, Any]] = []
    for feature in features:
        fg = safe_dict(feature.get("geo_bbox"))
        matched = False
        for existing in merged:
            eg = safe_dict(existing.get("geo_bbox"))
            if not fg or not eg:
                continue
            if cluster_family(existing) != cluster_family(feature):
                continue
            if feature_relation_bucket(existing) != feature_relation_bucket(feature):
                continue

            dist = geo_distance(fg, eg)
            overlap = overlap_ratio(fg, eg)
            if overlap >= 0.45 or dist <= 0.00012:
                merge_feature_pair(existing, feature)
                matched = True
                break

        if not matched:
            merged.append(dict(feature))

    return merged


def clean_feature_labels(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for feature in features:
        ftype = safe_str(feature.get("feature_type"), "")
        relation = safe_str(feature.get("location_relation"), "on_site")
        if ftype == "pond":
            feature["feature_id"] = "Pond" if relation == "on_site" else safe_str(feature.get("feature_id"), "Adjacent Pond")
        elif ftype == "former_pond":
            feature["feature_id"] = "Former Pond" if relation == "on_site" else safe_str(feature.get("feature_id"), "Adjacent Former Pond")
        elif ftype in ("disturbance", "fill_area"):
            feature["feature_id"] = "Disturbance Area"
        elif ftype == "possible_reclaimed_ground":
            feature["feature_id"] = "Possible Reclaimed Ground"
        elif ftype == "canal":
            feature["feature_id"] = "Adjacent Canal" if relation != "on_site" else "Canal"
        elif ftype == "drainage_feature":
            feature["feature_id"] = "Adjacent Creek / Drainage" if relation != "on_site" else "Drainage Feature"
        elif ftype == "beach_foreshore_or_coastal_edge":
            feature["feature_id"] = "Foreshore / Coastal Edge"
    return features


def should_merge_truth_features(f1: Dict[str, Any], f2: Dict[str, Any]) -> bool:
    b1 = safe_dict(f1.get("geo_bbox"))
    b2 = safe_dict(f2.get("geo_bbox"))
    if not b1 or not b2:
        return False
    if feature_relation_bucket(f1) != feature_relation_bucket(f2):
        return False

    fam1 = cluster_family(f1)
    fam2 = cluster_family(f2)
    if fam1 != fam2:
        return False

    overlap = overlap_ratio(b1, b2)
    dist = geo_distance(b1, b2)
    size_ratio = geo_bbox_size_ratio(b1, b2)

    if fam1 == "water":
        if overlap >= 0.22 and size_ratio >= 0.35:
            return True
        if overlap >= 0.08 and dist <= 0.00009 and size_ratio >= 0.35:
            return True
        if dist <= 0.000045 and size_ratio >= 0.55:
            return True
        return False

    if fam1 == "disturbance":
        return overlap >= 0.35 or (dist <= 0.00008 and size_ratio >= 0.45)

    return overlap >= 0.40


def dedupe_truth_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for feature in features:
        matched = False
        for existing in deduped:
            if should_merge_truth_features(existing, feature):
                merge_feature_pair(existing, feature)
                matched = True
                break
        if not matched:
            deduped.append(dict(feature))
    return deduped




def feature_has_historical_only_signature(feature: Dict[str, Any]) -> bool:
    labels = merge_string_lists(
        [safe_str(feature.get("primary_image_label"), ""), safe_str(feature.get("detected_on_image"), "")] +
        safe_list(feature.get("source_image_labels"))
    )
    years = [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
    text = feature_text_blob(feature)

    historical_seen = any(lbl.startswith("historical_") for lbl in labels) or any(y <= 2015 for y in years)
    current_year_seen = any(y >= 2020 for y in years)
    current_label_seen = any(lbl.startswith("current_") for lbl in labels)

    explicit_historical_only = any(
        k in text for k in [
            "1995 only",
            "historical imagery only",
            "visible in 1995 only",
            "visible in historic imagery only",
            "absent in current",
            "no current water",
            "no current water signature",
            "not visible in recent",
            "dry in recent",
            "disappeared",
            "no longer visible",
            "former pond",
            "former water",
            "historical only",
        ]
    )

    if not historical_seen:
        return False
    if explicit_historical_only:
        return True
    if current_year_seen or current_label_seen:
        return False
    return not feature_current_visible(feature)


def enforce_historical_secondary_former_lock(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    locked = [dict(f) for f in features]

    current_water_refs = [
        f for f in locked
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") in ("pond", "former_pond", "probable_pond")
        and feature_current_visible(f)
    ]

    for feature in locked:
        if safe_str(feature.get("location_relation"), "") != "on_site":
            continue

        ftype = safe_str(feature.get("feature_type"), "")
        if ftype not in ("pond", "former_pond", "probable_pond"):
            continue

        if feature_current_visible(feature):
            feature["feature_type"] = "pond"
            if safe_str(feature.get("risk_priority"), "") != "primary":
                feature["risk_priority"] = "secondary"
            continue

        if not feature_has_historical_only_signature(feature):
            continue

        fg = safe_dict(feature.get("geo_bbox"))
        same_as_current = False
        for current in current_water_refs:
            cg = safe_dict(current.get("geo_bbox"))
            if not fg or not cg:
                continue
            if overlap_ratio(fg, cg) >= 0.62 and geo_bbox_size_ratio(fg, cg) >= 0.45:
                same_as_current = True
                break

        if same_as_current:
            continue

        feature["feature_type"] = "former_pond"
        feature["risk_priority"] = "primary"
        feature["preserved_by_rule"] = True

    return locked

def extract_locked_former_ponds(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    locked: List[Dict[str, Any]] = []

    for feature in [dict(f) for f in safe_list(features) if isinstance(f, dict)]:
        relation = safe_str(feature.get("location_relation"), "")
        ftype = safe_str(feature.get("feature_type"), "")
        if relation != "on_site":
            continue

        keep_as_former = False
        if ftype == "former_pond":
            keep_as_former = True
        elif feature.get("preserved_by_rule"):
            keep_as_former = True
        elif ftype == "probable_pond" and feature_has_historical_only_signature(feature):
            keep_as_former = True

        if not keep_as_former:
            continue

        promoted = dict(feature)
        promoted["feature_type"] = "former_pond"
        promoted["risk_priority"] = "primary"
        promoted["preserved_by_rule"] = True
        locked.append(promoted)

    deduped: List[Dict[str, Any]] = []
    for feature in locked:
        matched = False
        for existing in deduped:
            if should_merge_truth_features(existing, feature):
                merge_feature_pair(existing, feature)
                existing["feature_type"] = "former_pond"
                existing["risk_priority"] = "primary"
                existing["preserved_by_rule"] = True
                matched = True
                break
        if not matched:
            deduped.append(dict(feature))

    return deduped


def preserve_primary_historical_water_features(
    primary_features: List[Dict[str, Any]],
    merged_features: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    protected = extract_locked_former_ponds(primary_features)
    if not protected:
        return [dict(f) for f in merged_features]

    out = [dict(f) for f in merged_features]
    for feature in protected:
        matched = False
        for existing in out:
            if should_merge_truth_features(existing, feature):
                if safe_str(existing.get("feature_type"), "") == "pond" and safe_str(feature.get("feature_type"), "") == "former_pond":
                    continue
                merge_feature_pair(existing, feature)
                existing["feature_type"] = "former_pond"
                existing["risk_priority"] = "primary"
                existing["preserved_by_rule"] = True
                matched = True
                break
        if not matched:
            out.append(dict(feature))
    return out


def build_truth_flags_from_features(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_features = [dict(f) for f in features if isinstance(f, dict)]
    truth_features = dedupe_truth_features(raw_features)

    forced_former_ponds = extract_locked_former_ponds(raw_features)
    for feature in forced_former_ponds:
        matched = False
        for existing in truth_features:
            if should_merge_truth_features(existing, feature):
                if safe_str(existing.get("feature_type"), "") == "pond":
                    continue
                merge_feature_pair(existing, feature)
                existing["feature_type"] = "former_pond"
                existing["risk_priority"] = "primary"
                existing["preserved_by_rule"] = True
                matched = True
                break
        if not matched:
            truth_features.append(dict(feature))

    on_site = [f for f in truth_features if safe_str(f.get("location_relation"), "") == "on_site"]
    adjacent = [f for f in truth_features if safe_str(f.get("location_relation"), "") != "on_site"]

    current_ponds = [f for f in on_site if safe_str(f.get("feature_type"), "") == "pond"]
    former_ponds = [f for f in on_site if safe_str(f.get("feature_type"), "") == "former_pond"]
    probable_ponds = [f for f in on_site if safe_str(f.get("feature_type"), "") == "probable_pond"]
    disturbances = [f for f in on_site if safe_str(f.get("feature_type"), "") in ("disturbance", "fill_area", "retaining_or_cut_fill")]
    structures = [f for f in on_site if safe_str(f.get("feature_type"), "") in ("existing_structure", "former_structure", "hardstand_or_slab")]
    reclaimed_ground = [f for f in on_site if safe_str(f.get("feature_type"), "") == "possible_reclaimed_ground"]
    vegetation = [f for f in on_site if safe_str(f.get("feature_type"), "") == "significant_tree_or_vegetation"]

    canals = [f for f in truth_features if safe_str(f.get("feature_type"), "") == "canal"]
    drainage_features = [f for f in truth_features if safe_str(f.get("feature_type"), "") == "drainage_feature"]
    coastal_features = [f for f in truth_features if safe_str(f.get("feature_type"), "") == "beach_foreshore_or_coastal_edge"]
    external_waterbodies = [f for f in truth_features if safe_str(f.get("feature_type"), "") == "large_external_waterbody"]
    uncertain_water = [f for f in truth_features if safe_str(f.get("feature_type"), "") == "uncertain_water_related_feature"]

    water_context_types = (
        "pond", "former_pond", "probable_pond", "drainage_feature", "canal",
        "beach_foreshore_or_coastal_edge", "large_external_waterbody",
        "uncertain_water_related_feature"
    )
    adjacent_water = [f for f in adjacent if safe_str(f.get("feature_type"), "") in water_context_types]
    any_water_context = [f for f in truth_features if safe_str(f.get("feature_type"), "") in water_context_types]

    return {
        "truth_features": truth_features,
        "current_ponds": current_ponds,
        "former_ponds": former_ponds,
        "probable_ponds": probable_ponds,
        "disturbances": disturbances,
        "structures": structures,
        "reclaimed_ground": reclaimed_ground,
        "vegetation": vegetation,
        "canals": canals,
        "drainage_features": drainage_features,
        "coastal_features": coastal_features,
        "external_waterbodies": external_waterbodies,
        "uncertain_water": uncertain_water,
        "adjacent_water": adjacent_water,
        "any_water_context": any_water_context,
        "has_current_pond": bool(current_ponds),
        "has_former_pond": bool(former_ponds),
        "has_probable_pond": bool(probable_ponds),
        "has_disturbance": bool(disturbances),
        "has_structures": bool(structures),
        "has_reclaimed_ground": bool(reclaimed_ground),
        "has_vegetation": bool(vegetation),
        "has_canal": bool(canals),
        "has_drainage_feature": bool(drainage_features),
        "has_coastal_feature": bool(coastal_features),
        "has_external_waterbody": bool(external_waterbodies),
        "has_uncertain_water": bool(uncertain_water),
        "has_adjacent_water": bool(adjacent_water),
        "has_any_water_context": bool(any_water_context),
        "current_pond_count": len(current_ponds),
        "former_pond_count": len(former_ponds),
        "probable_pond_count": len(probable_ponds),
        "disturbance_count": len(disturbances),
        "structure_count": len(structures),
        "reclaimed_ground_count": len(reclaimed_ground),
        "canal_count": len(canals),
        "drainage_feature_count": len(drainage_features),
        "coastal_feature_count": len(coastal_features),
        "external_waterbody_count": len(external_waterbodies),
        "adjacent_water_count": len(adjacent_water),
    }

def build_truth_layer_from_features(features: List[Dict[str, Any]]) -> Dict[str, str]:
    flags = build_truth_flags_from_features(features)

    current_ponds = flags["current_ponds"]
    former_ponds = flags["former_ponds"]
    probable_ponds = flags["probable_ponds"]
    disturbances = flags["disturbances"]
    structures = flags.get("structures", [])
    reclaimed_ground = flags.get("reclaimed_ground", [])
    canals = flags.get("canals", [])
    drainage_features = flags.get("drainage_features", [])
    coastal_features = flags.get("coastal_features", [])
    external_waterbodies = flags.get("external_waterbodies", [])
    uncertain_water = flags.get("uncertain_water", [])
    adjacent_water = flags["adjacent_water"]

    water_context_parts = []
    if canals:
        water_context_parts.append("adjacent canal / engineered waterway context")
    if drainage_features:
        water_context_parts.append("creek or drainage context")
    if coastal_features:
        water_context_parts.append("foreshore / coastal water context")
    if external_waterbodies:
        water_context_parts.append("external waterbody context")
    if uncertain_water:
        water_context_parts.append("uncertain water-related context")

    has_water_context = bool(water_context_parts or adjacent_water)

    if current_ponds and former_ponds:
        summary = "A current pond is present on-site. Historical imagery also indicates at least one additional former or altered pond / wet depression on-site."
    elif current_ponds and probable_ponds:
        summary = "A current pond is present on-site. Historical imagery also suggests one or more additional secondary pond / wet depression signatures on-site."
    elif former_ponds:
        summary = "No current isolated on-site pond is clearly visible; however, historical imagery indicates former ponds or infilled wet depressions on-site."
    elif current_ponds:
        summary = "A current isolated pond feature is present on-site."
    elif probable_ponds:
        summary = "Possible former pond or wet depression signatures are present on-site in historical imagery."
    elif has_water_context:
        summary = "No isolated on-site pond was carried through the final interpretation; however, nearby water context was identified and should be considered for abnormal moisture assessment."
    else:
        summary = "No on-site or adjacent water context was carried through the final interpretation from available imagery."

    if reclaimed_ground:
        summary += " Possible reclaimed or canal-edge fill indicators are also visible on-site."
    if structures:
        summary += " Existing on-site building / development is visible."
    elif disturbances:
        summary += " Site disturbance or possible fill-related ground modification is also visible."
    if water_context_parts:
        summary += " " + "; ".join(water_context_parts).capitalize() + " was identified and treated as geotechnically relevant context rather than an isolated pond."
    elif adjacent_water:
        summary += " Adjacent off-site water features were identified and treated as geotechnically relevant context."

    if current_ponds:
        if len(current_ponds) == 1 and former_ponds:
            on_site_summary = (
                "1 current isolated on-site pond feature is present. "
                f"{len(former_ponds)} additional former pond feature{'s' if len(former_ponds) != 1 else ''} "
                f"{'are' if len(former_ponds) != 1 else 'is'} also identified from historical imagery."
            )
        elif len(current_ponds) == 1:
            on_site_summary = "1 current isolated on-site pond feature is present in the final interpretation."
        else:
            on_site_summary = f"{len(current_ponds)} distinct current isolated on-site pond features are present in the final interpretation."
            if former_ponds:
                on_site_summary += (
                    f" {len(former_ponds)} additional former pond feature{'s' if len(former_ponds) != 1 else ''} "
                    f"{'are' if len(former_ponds) != 1 else 'is'} also identified from historical imagery."
                )
    elif former_ponds:
        on_site_summary = f"No current isolated on-site pond is clearly visible. Historical imagery indicates {len(former_ponds)} former pond feature{'s' if len(former_ponds) != 1 else ''} or altered wet depressions within the site boundary."
    elif probable_ponds:
        on_site_summary = "No current isolated on-site pond is clearly visible. One or more secondary pond / wet depression signatures remain uncertain and should be treated conservatively."
    elif reclaimed_ground:
        on_site_summary = "No isolated on-site pond was carried through; however, possible reclaimed or canal-edge fill indicators are visible on-site."
    else:
        on_site_summary = "No isolated on-site pond was carried through the final interpretation."

    if structures:
        on_site_summary += " Existing on-site building / development is visible."
    elif disturbances:
        on_site_summary += " Disturbance or possible fill-related ground modification is also visible on-site."

    if canals:
        adjacent_summary = f"{len(canals)} adjacent canal / engineered waterway feature{'s were' if len(canals) != 1 else ' was'} identified and treated as abnormal moisture / canal-edge context, not as an isolated pond."
    elif drainage_features:
        adjacent_summary = f"{len(drainage_features)} creek or drainage feature{'s were' if len(drainage_features) != 1 else ' was'} identified and treated as abnormal moisture context."
    elif coastal_features:
        adjacent_summary = f"{len(coastal_features)} foreshore or coastal water context feature{'s were' if len(coastal_features) != 1 else ' was'} identified and treated as abnormal moisture context."
    elif external_waterbodies:
        adjacent_summary = f"{len(external_waterbodies)} nearby external waterbody feature{'s were' if len(external_waterbodies) != 1 else ' was'} identified and treated as abnormal moisture context."
    elif adjacent_water:
        adjacent_summary = f"{len(adjacent_water)} adjacent water feature{'s were' if len(adjacent_water) != 1 else ' was'} identified outside the site boundary and treated as abnormal moisture context."
    else:
        adjacent_summary = "No significant adjacent water features were carried through the final interpretation."

    if former_ponds and current_ponds:
        screening = "A current pond is present on-site and historical imagery also indicates at least one additional former or altered pond / wet depression. Detailed geotechnical investigation is strongly recommended."
    elif former_ponds or probable_ponds:
        screening = "Historical imagery indicates former or possible former pond / wet depression signatures on-site. Detailed geotechnical investigation is strongly recommended."
    elif current_ponds:
        screening = "A current isolated on-site water feature is present and should be considered in geotechnical assessment. Detailed geotechnical investigation is strongly recommended."
    elif reclaimed_ground or canals:
        screening = "No isolated on-site pond was carried through; however, adjacent canal / waterway context and possible canal-edge or reclaimed-ground conditions are relevant to abnormal moisture and fill assessment. Detailed geotechnical investigation is strongly recommended."
    elif drainage_features or coastal_features or external_waterbodies or adjacent_water:
        screening = "Nearby water context is relevant to abnormal moisture assessment under AS2870-style investigation planning. Detailed geotechnical investigation is strongly recommended."
    elif disturbances:
        screening = "No isolated on-site pond was carried through the final interpretation; however, local disturbance or possible fill-related ground modification is visible. Detailed geotechnical investigation is strongly recommended."
    elif structures:
        screening = "Existing on-site building / development is visible. No significant water or disturbance indicators were carried through the final interpretation from available imagery. A detailed geotechnical investigation is recommended to confirm ground conditions and assess their suitability for residential foundation design in accordance with AS2870."
    else:
        screening = "No significant water or disturbance indicators were carried through the final interpretation from available imagery. Absence of visible indicators does not confirm appropriate or uniform ground conditions, A detailed geotechnical investigation is recommended to confirm ground conditions and assess their suitability for residential foundation design in accordance with AS2870."

    return {
        "summary": dedupe_sentences(summary),
        "on_site_summary": dedupe_sentences(on_site_summary),
        "adjacent_context_summary": dedupe_sentences(adjacent_summary),
        "screening_outcome": dedupe_sentences(screening),
    }

def build_standard_geotechnical_risks(features: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    flags = build_truth_flags_from_features(features)
    risks: List[Dict[str, str]] = []

    disturbance_text_blob = " ".join(
        [
            safe_str(f.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(f.get("evidence"))])
            for f in flags.get("disturbances", [])
        ]
    ).lower()
    structure_text_blob = " ".join(
        [
            safe_str(f.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(f.get("evidence"))])
            for f in flags.get("structures", [])
        ]
    ).lower()
    all_on_site_blob = " ".join(
        safe_str(f.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(f.get("evidence"))])
        for f in flags.get("truth_features", [])
        if safe_str(f.get("location_relation"), "") == "on_site"
    ).lower()
    structure_risk_text = structure_text_blob + " " + disturbance_text_blob
    structure_risk_signal = any(k in structure_risk_text for k in [
        "former", "removed", "demolished", "demolition", "hardstand", "slab",
        "driveway", "pavement", "service trench", "regrading", "reworked",
        "earthworks", "cut", "fill", "retaining", "platform preparation"
    ])
    building_related_disturbance = (
        has_structure_or_hardstand_signal(disturbance_text_blob)
        or structure_risk_signal
    )
    has_embedded_disturbance_signal = has_strong_disturbance_signal(disturbance_text_blob)

    if flags.get("has_disturbance") or building_related_disturbance or has_embedded_disturbance_signal:
        risks.append({
            "title": "Disturbance / Fill Risk",
            "level": "ELEVATED",
            "text": (
                "Visible earthworks or disturbance within the site suggests that ground modification has occurred. "
                "This may include cut, fill, or a combination of both associated with platform preparation. "
                "The presence, extent, origin and composition of any placed fill cannot be confirmed from imagery alone. "
                "Reworked ground may result in variable strength, moisture conditions and overall soil behaviour, which may impact foundation "
                "performance and footing type if not properly assessed in accordance with AS2870."
                if not building_related_disturbance else
                "Existing or former building / hardstand footprints and associated site disturbance indicate prior localised ground modification may have occurred. "
                "These areas can be associated with regrading, hardstand construction, service installation, slab preparation, demolition, or localised cut/fill works. "
                "The presence and extent of any placed fill cannot be confirmed from imagery alone. Modified or reworked ground may contribute to variable near-surface conditions, "
                "foundation performance issues, or differential movement if not properly assessed in accordance with AS2870."
            ),
        })

    if flags.get("has_reclaimed_ground"):
        risks.append({
            "title": "Possible Reclaimed / Canal-Edge Fill",
            "level": "HIGH",
            "text": (
                "Visual indicators suggest possible reclaimed ground or canal-edge fill within the site. "
                "Canal-front and reclaimed-waterway settings may include placed fill, dredged material, reworked sands, soft compressible soils, "
                "variable founding conditions and groundwater influence. These conditions can be difficult to classify from imagery alone and should be "
                "specifically assessed during detailed geotechnical investigation for AS2870 site classification."
            ),
        })

    if flags.get("has_former_pond") or flags.get("has_probable_pond"):
        risks.append({
            "title": "Historical Pond / Fill Risk",
            "level": "HIGH",
            "text": (
                "Historical imagery indicates the presence of former water features or wet depressions within the site. "
                "These areas may have been infilled over time, potentially with uncontrolled or undocumented materials. "
                "Infilled pond zones can present significant geotechnical risks, including compressible soils, variable founding conditions, "
                "abnormal moisture conditions and potential for differential movement. The nature and extent of any fill materials should be confirmed through detailed "
                "geotechnical investigation in accordance with AS2870 requirements."
            ),
        })

    if flags.get("has_current_pond"):
        risks.append({
            "title": "Pond / Wet Area Risk",
            "level": "HIGH",
            "text": (
                "The presence of a current pond or pond-like depression within the site indicates localised moisture influence and potential "
                "variability in ground conditions. Such features are commonly associated with elevated moisture contents, soft or "
                "compressible soils, and conditions that may contribute to ground movement. These areas may remain seasonally wet or act "
                "as surface water collection points, which can impact foundation performance and are relevant to abnormal moisture and site classification "
                "considerations in accordance with AS2870."
            ),
        })

    if flags.get("has_canal"):
        risks.append({
            "title": "Adjacent Canal / Waterway Context",
            "level": "MODERATE",
            "text": (
                "An adjacent canal or engineered linear waterway has been identified outside the site boundary. "
                "This has not been interpreted as an isolated on-site pond; however, any adjacent water setting may still be relevant to abnormal moisture "
                "conditions, groundwater influence, reclaimed/canal-edge fill history, and geotechnical investigation planning under AS2870."
            ),
        })

    if flags.get("has_drainage_feature"):
        risks.append({
            "title": "Creek / Drainage Context",
            "level": "MODERATE",
            "text": (
                "A nearby creek, drainage line or flow path has been identified. Such features may indicate local moisture variation, overland flow paths, "
                "alluvial or soft sediments, and abnormal moisture conditions relevant to AS2870 investigation planning."
            ),
        })

    if flags.get("has_coastal_feature"):
        risks.append({
            "title": "Foreshore / Coastal Context",
            "level": "MODERATE",
            "text": (
                "A coastal, foreshore, tidal or dune-related setting has been identified. These settings may include loose to variable sands, shallow groundwater, "
                "marine or estuarine deposits, erosion susceptibility and abnormal moisture influence. These factors should be considered during detailed "
                "geotechnical investigation and AS2870 site classification."
            ),
        })

    if flags.get("has_external_waterbody") and not (flags.get("has_canal") or flags.get("has_coastal_feature") or flags.get("has_drainage_feature")):
        risks.append({
            "title": "Adjacent Waterbody Context",
            "level": "MODERATE",
            "text": (
                "A significant external waterbody has been identified near the site. While not interpreted as an on-site pond, nearby water may still be relevant "
                "to groundwater, surface drainage, moisture variation and abnormal moisture considerations for AS2870 investigation planning."
            ),
        })

    return risks

def rebuild_findings_notes_from_features(features: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    flags = build_truth_flags_from_features(features)

    current_ponds = flags["current_ponds"]
    former_ponds = flags["former_ponds"]
    probable_ponds = flags["probable_ponds"]
    disturbances = flags["disturbances"]

    # IMPORTANT:
    # "Former Water Features" must only describe former / probable former on-site evidence.
    # It must never reuse current-pond wording.
    if former_ponds:
        pond_status = "strong_evidence"
        pond_conf = "high"
        if current_ponds:
            pond_note = "Historical imagery indicates at least one additional former pond or altered wet depression on-site, separate from the current pond."
        else:
            pond_note = "Historical imagery indicates former pond or altered wet depression signatures on-site."
    elif probable_ponds:
        pond_status = "possible"
        pond_conf = "medium"
        pond_note = "Possible former pond or wet depression signatures are present on-site in historical imagery."
    else:
        pond_status = "none"
        pond_conf = "medium"
        pond_note = "No on-site former water features were carried through the final interpretation."

    disturbance_text_blob = " ".join(
        [
            safe_str(d.get("notes"), "")
            + " "
            + " ".join([str(x) for x in safe_list(d.get("evidence"))])
            for d in disturbances
        ]
    ).lower()

    all_on_site = [
        f for f in flags.get("truth_features", [])
        if safe_str(f.get("location_relation"), "") == "on_site"
    ]
    full_evidence_blob = " ".join(
        safe_str(f.get("notes"), "") + " " +
        " ".join([str(x) for x in safe_list(f.get("evidence"))])
        for f in all_on_site
    ).lower()

    structure_risk_signal = any(k in full_evidence_blob for k in [
        "former structure", "former dwelling", "removed structure", "demolished",
        "demolition", "hardstand", "slab", "driveway", "pavement",
        "service trench", "regrading", "reworked", "earthworks", "cut",
        "fill", "retaining", "platform preparation"
    ])
    has_building_disturbance = has_structure_or_hardstand_signal(disturbance_text_blob) or structure_risk_signal
    has_disturbance_signal = has_strong_disturbance_signal(disturbance_text_blob)

    if disturbances or has_building_disturbance or has_disturbance_signal:
        if has_building_disturbance:
            fill_status = "possible"
            fill_conf = "medium"
            fill_note = "Former structures, hardstand, slabs, driveways or associated on-site disturbance are visible, indicating prior localised ground modification. Cut/fill conditions cannot be confirmed from imagery alone."
        elif disturbances or has_disturbance_signal:
            fill_status = "possible"
            fill_conf = "medium"
            fill_note = "Recent on-site earthworks or reworked ground are visible. This may include cut, fill, or a combination of both; the presence of placed fill cannot be confirmed from imagery alone."
        else:
            fill_status = "possible"
            fill_conf = "medium"
            fill_note = "Minor disturbance indicators are visible on-site."
    else:
        fill_status = "none"
        fill_conf = "medium"
        fill_note = "No clear on-site disturbance or fill-related ground modification was carried through the final interpretation."

    return {
        "former_ponds_dams": {
            "status": pond_status,
            "confidence": pond_conf,
            "notes": pond_note,
        },
        "fill_or_disturbance": {
            "status": fill_status,
            "confidence": fill_conf,
            "notes": fill_note,
        },
    }



def is_historical_secondary_former_candidate(feature: Dict[str, Any]) -> bool:
    ftype = safe_str(feature.get("feature_type"), "")
    relation = safe_str(feature.get("location_relation"), "")
    conf = safe_str(feature.get("confidence"), "low")
    years = feature_year_set(feature)
    text = (safe_str(feature.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(feature.get("evidence"))])).lower()

    if relation != "on_site":
        return False
    if ftype not in ("former_pond", "probable_pond"):
        return False
    if conf not in ("medium", "high"):
        return False
    if not years:
        return False
    if not any(y <= 2015 for y in years):
        return False
    if any(k in text for k in ["adjacent", "outside boundary", "off-site", "off site", "contextual"]):
        return False
    return True


def collect_force_promoted_former_ponds(
    analyses: List[Optional[Dict[str, Any]]],
    resolved: Optional[Dict[str, Any]],
    image_lookup: Optional[Dict[str, Dict[str, Any]]],
    final_features: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    promotable: List[Dict[str, Any]] = []

    current_on_site = [
        f for f in safe_list(final_features)
        if isinstance(f, dict)
        and safe_str(f.get("feature_type"), "") == "pond"
        and safe_str(f.get("location_relation"), "") == "on_site"
    ]

    for analysis in analyses:
        if not isinstance(analysis, dict):
            continue

        raw_features: List[Dict[str, Any]] = []
        raw_features.extend([dict(f) for f in safe_list(analysis.get("distinct_features")) if isinstance(f, dict)])

        candidate_features = build_features_from_candidates(
            [c for c in safe_list(analysis.get("candidates")) if isinstance(c, dict)]
        )
        candidate_features = upgrade_geotech_features(candidate_features)
        raw_features.extend(candidate_features)

        if resolved and image_lookup:
            raw_features = finalize_feature_geometry_and_ids(raw_features, resolved=resolved, image_lookup=image_lookup)

        for feature in raw_features:
            if not is_historical_secondary_former_candidate(feature):
                continue

            fg = safe_dict(feature.get("geo_bbox"))
            if not fg:
                continue

            # Keep secondary historical ponds unless they are basically the same geometry as the main current pond.
            same_as_primary = False
            for existing in current_on_site:
                eg = safe_dict(existing.get("geo_bbox"))
                if not eg:
                    continue
                if overlap_ratio(fg, eg) >= 0.62 and geo_bbox_size_ratio(fg, eg) >= 0.45:
                    same_as_primary = True
                    break
            if same_as_primary:
                continue

            promoted = dict(feature)
            promoted["feature_type"] = "former_pond"
            promoted["risk_priority"] = "primary"
            promoted["preserved_by_rule"] = True
            promotable.append(promoted)

    deduped: List[Dict[str, Any]] = []
    for feature in promotable:
        matched = False
        for existing in deduped:
            if should_merge_truth_features(existing, feature):
                merge_feature_pair(existing, feature)
                existing["feature_type"] = "former_pond"
                existing["preserved_by_rule"] = True
                matched = True
                break
        if not matched:
            deduped.append(dict(feature))

    return deduped


def inject_force_promoted_former_ponds(
    final_analysis: Dict[str, Any],
    analyses: List[Optional[Dict[str, Any]]],
    resolved: Optional[Dict[str, Any]],
    image_lookup: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    updated = dict(final_analysis)
    final_features = [dict(f) for f in safe_list(updated.get("distinct_features")) if isinstance(f, dict)]
    promoted = collect_force_promoted_former_ponds(analyses, resolved, image_lookup, final_features)

    if not promoted:
        return updated

    for feature in promoted:
        matched = False
        for existing in final_features:
            if should_merge_truth_features(existing, feature):
                # If an existing on-site current pond occupies nearly the same footprint, do not collapse away the former pond.
                if safe_str(existing.get("feature_type"), "") == "pond" and safe_str(feature.get("feature_type"), "") == "former_pond":
                    continue
                merge_feature_pair(existing, feature)
                matched = True
                break
        if not matched:
            final_features.append(dict(feature))

    updated["distinct_features"] = final_features
    return updated

def analysis_has_on_site_former_water_evidence(analysis: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(analysis, dict):
        return False

    findings = ensure_findings_dict(analysis.get("historical_findings"))
    former = safe_dict(findings.get("former_ponds_dams"))
    status = safe_str(former.get("status"), "none").lower()
    notes = safe_str(former.get("notes"), "").lower()

    if status in ("likely", "strong_evidence"):
        if not any(k in notes for k in ["adjacent", "outside boundary", "off-site", "off site", "contextual"]):
            return True

    for feature in [f for f in safe_list(analysis.get("distinct_features")) if isinstance(f, dict)]:
        if safe_str(feature.get("location_relation"), "") != "on_site":
            continue
        ftype = safe_str(feature.get("feature_type"), "")
        if ftype in ("former_pond", "probable_pond") and feature_has_historical_only_signature(feature):
            return True

        blob = feature_text_blob(feature)
        years = [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
        if (
            ftype in ("pond", "former_pond", "probable_pond")
            and any(y <= 2015 for y in years)
            and any(k in blob for k in ["former", "historical", "infilled", "altered wet depression", "secondary pond", "dry in recent", "no longer visible"])
        ):
            return True

    return False


def strongest_former_water_note(analyses: List[Optional[Dict[str, Any]]]) -> str:
    best_note = ""
    best_rank = -1
    rank_map = {"none": 0, "possible": 1, "likely": 2, "strong_evidence": 3}

    for analysis in analyses:
        if not isinstance(analysis, dict):
            continue
        findings = ensure_findings_dict(analysis.get("historical_findings"))
        former = safe_dict(findings.get("former_ponds_dams"))
        note = safe_str(former.get("notes"), "").strip()
        status = safe_str(former.get("status"), "none").lower()
        rank = rank_map.get(status, 0)
        if note and rank > best_rank:
            best_note = note
            best_rank = rank

    return best_note


def ensure_historical_pond_risk_present(geotechnical_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    risks = [dict(r) for r in safe_list(geotechnical_risks) if isinstance(r, dict)]
    if any(safe_str(r.get("title"), "") == "Historical Pond / Fill Risk" for r in risks):
        return risks

    risks.append({
        "title": "Historical Pond / Fill Risk",
        "level": "HIGH",
        "text": (
            "Historical imagery indicates the presence of former water features or wet depressions within the site. "
            "These areas may have been infilled over time, potentially with uncontrolled or undocumented materials. "
            "Infilled pond zones can present significant geotechnical risks, including compressible soils, variable founding conditions, "
            "and potential for differential movement. The nature and extent of any fill materials should be confirmed through detailed "
            "geotechnical investigation in accordance with AS2870 requirements."
        ),
    })
    return risks


def enforce_former_water_truth_lock(
    final_analysis: Dict[str, Any],
    analyses: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    updated = dict(final_analysis)

    if not any(analysis_has_on_site_former_water_evidence(a) for a in analyses):
        return updated

    features = [dict(f) for f in safe_list(updated.get("distinct_features")) if isinstance(f, dict)]
    current_on_site = [
        f for f in features
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") == "pond"
    ]
    former_on_site = [
        f for f in features
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") == "former_pond"
    ]

    if not former_on_site:
        features.append({
            "feature_id": "Former Pond",
            "feature_type": "former_pond",
            "location_relation": "on_site",
            "confidence": "medium",
            "notes": strongest_former_water_note(analyses) or (
                "Historical imagery indicates at least one additional former pond or altered wet depression on-site."
                if current_on_site else
                "Historical imagery indicates former pond or altered wet depression signatures on-site."
            ),
            "evidence": ["historical on-site former pond truth lock"],
            "detected_in_years": [],
            "primary_image_label": "",
            "detected_on_image": "",
            "approximate_bbox_norm": [],
            "risk_priority": "primary",
            "preserved_by_rule": True,
        })
        updated["distinct_features"] = features

    findings = ensure_findings_dict(updated.get("historical_findings"))
    note = strongest_former_water_note(analyses) or (
        "Historical imagery indicates at least one additional former pond or altered wet depression on-site, separate from the current pond."
        if current_on_site else
        "Historical imagery indicates former pond or altered wet depression signatures on-site."
    )
    findings["former_ponds_dams"] = {
        "status": "strong_evidence",
        "confidence": "high",
        "notes": note,
    }
    updated["historical_findings"] = findings

    has_disturbance = any(
        safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") in ("disturbance", "fill_area")
        for f in features
    )
    has_adjacent = any(
        safe_str(f.get("location_relation"), "") != "on_site"
        and safe_str(f.get("feature_type"), "") in ("pond", "former_pond", "probable_pond", "drainage_feature")
        for f in features
    )

    if current_on_site:
        summary = "A current pond is present on-site. Historical imagery also indicates at least one additional former or altered pond / wet depression on-site."
        on_site_summary = "1 current on-site pond feature is present. 1 additional former pond feature is also identified from historical imagery."
        screening = "A current pond is present on-site and historical imagery also indicates at least one additional former or altered pond / wet depression. Detailed geotechnical investigation is strongly recommended."
    else:
        summary = "No current on-site water bodies are clearly visible; however, historical imagery indicates former ponds or infilled wet depressions on-site."
        on_site_summary = "No current on-site water bodies are clearly visible. Historical imagery indicates former pond features or altered wet depressions within the site boundary."
        screening = "Historical imagery indicates former or possible former pond / wet depression signatures on-site. Detailed geotechnical investigation is strongly recommended."

    if has_disturbance:
        summary += " Site disturbance or possible fill-related ground modification is also visible."
        on_site_summary += " Disturbance or possible fill-related ground modification is also visible on-site."
    if has_adjacent:
        summary += " Adjacent off-site water features were identified and treated as contextual only."
        adjacent_context_summary = "Adjacent water features were identified outside the site boundary and treated as contextual only."
    else:
        adjacent_context_summary = "No significant adjacent water features were carried through the final interpretation."

    updated["summary"] = dedupe_sentences(summary)
    updated["on_site_summary"] = dedupe_sentences(on_site_summary)
    updated["adjacent_context_summary"] = dedupe_sentences(adjacent_context_summary)
    updated["screening_outcome"] = dedupe_sentences(screening)
    updated["geotechnical_risks"] = ensure_historical_pond_risk_present(updated.get("geotechnical_risks", []))

    return updated


def sanitize_analysis_for_report(
    analysis: Dict[str, Any],
    resolved: Optional[Dict[str, Any]] = None,
    image_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cleaned = dict(analysis)
    cleaned["historical_findings"] = ensure_findings_dict(cleaned.get("historical_findings"))
    features = [f for f in safe_list(cleaned.get("distinct_features")) if isinstance(f, dict)]

    if resolved and image_lookup:
        rebuilt = []
        polygon = resolved.get("polygon")
        for f in features:
            f2 = dict(f)
            f2["source_image_labels"] = merge_string_lists([safe_str(f2.get("primary_image_label"), "")] + safe_list(f2.get("source_image_labels")))
            if safe_dict(f2.get("geo_bbox")) and polygon:
                f2["location_relation"] = resolve_feature_relation(
                    safe_dict(f2["geo_bbox"]),
                    polygon,
                    feature_type=safe_str(f2.get("feature_type"), ""),
                    notes=safe_str(f2.get("notes"), ""),
                )
            rebuilt.append(f2)

        features = finalize_feature_geometry_and_ids(rebuilt, resolved=resolved, image_lookup=image_lookup)

    # Conservative final cleanup only; avoids touching upstream imagery / AI logic.
    features = final_consolidate_features([dict(f) for f in features])
    features = enforce_historical_secondary_former_lock([dict(f) for f in features])
    features = assign_stable_feature_ids([dict(f) for f in features])
    features = clean_feature_labels([dict(f) for f in features])
    cleaned["distinct_features"] = features
    cleaned["feature_timeline"] = normalize_feature_timeline_items(cleaned.get("feature_timeline", []))
    cleaned["change_timeline"] = normalize_change_timeline_items(cleaned.get("change_timeline", []))
    cleaned["visible_observations"] = [str(x) for x in safe_list(cleaned.get("visible_observations")) if str(x).strip()]
    cleaned["possible_risks"] = [str(x) for x in safe_list(cleaned.get("possible_risks")) if str(x).strip()]
    cleaned["limitations"] = dedupe_limitations(cleaned.get("limitations", []))
    cleaned["recommended_investigation_focus"] = []
    cleaned["recommended_follow_up"] = [
        "Detailed geotechnical investigation is strongly recommended based on identified site history risks."
    ]

    truth = build_truth_layer_from_features(features)
    cleaned["summary"] = truth["summary"]
    cleaned["on_site_summary"] = truth["on_site_summary"]
    cleaned["adjacent_context_summary"] = truth["adjacent_context_summary"]
    cleaned["screening_outcome"] = truth["screening_outcome"]
    cleaned["geotechnical_risks"] = build_standard_geotechnical_risks(features)

    rebuilt_findings = rebuild_findings_notes_from_features(features)
    cleaned["historical_findings"]["former_ponds_dams"] = rebuilt_findings["former_ponds_dams"]
    cleaned["historical_findings"]["fill_or_disturbance"] = rebuilt_findings["fill_or_disturbance"]

    cleaned["visible_observations"] = merge_string_lists(cleaned.get("visible_observations", []))
    cleaned["possible_risks"] = merge_string_lists(cleaned.get("possible_risks", []))
    cleaned["limitations"] = dedupe_limitations(cleaned.get("limitations", []))
    return cleaned


def geocode_address_mapbox(address: str) -> Optional[Dict[str, Any]]:
    if not MAPBOX_TOKEN:
        return None

    url = f"{MAPBOX_GEOCODE_URL}/{requests.utils.quote(address)}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "country": "au",
        "limit": 1,
        "autocomplete": "false",
    }

    data = safe_get(url, params)
    if not data:
        return None

    features = data.get("features", [])
    if not features:
        return None

    feature = features[0]
    center = feature.get("center", [])
    if len(center) != 2:
        return None

    lng, lat = center[0], center[1]
    return {
        "address_matched": feature.get("place_name"),
        "lat": lat,
        "lng": lng,
        "relevance": feature.get("relevance"),
    }


def resolve_site_geometry(payload: SiteRequest) -> Dict[str, Any]:
    if payload.polygon:
        polygon = ensure_closed_polygon(payload.polygon)
        centroid = polygon_centroid(polygon)
        raw_bbox = polygon_bbox(polygon)
        bbox = expand_bbox_meters(raw_bbox, center_lat=centroid["lat"], pad_m=POLYGON_PAD_M)
        context_bbox = expand_bbox_meters(raw_bbox, center_lat=centroid["lat"], pad_m=CONTEXT_PAD_M)
        wide_context_bbox = expand_bbox_meters(raw_bbox, center_lat=centroid["lat"], pad_m=WIDE_CONTEXT_PAD_M)
        subzones = split_bbox_into_subzones(bbox, centroid["lat"])

        dims = bbox_width_height_m(bbox, centroid["lat"])
        context_dims = bbox_width_height_m(context_bbox, centroid["lat"])
        wide_dims = bbox_width_height_m(wide_context_bbox, centroid["lat"])

        return {
            "lat": centroid["lat"],
            "lng": centroid["lng"],
            "location_source": "polygon",
            "matched_address": None,
            "polygon": polygon,
            "bbox": bbox,
            "context_bbox": context_bbox,
            "wide_context_bbox": wide_context_bbox,
            "subzones": subzones,
            "bbox_width_m": round(dims["width_m"], 1),
            "bbox_height_m": round(dims["height_m"], 1),
            "context_bbox_width_m": round(context_dims["width_m"], 1),
            "context_bbox_height_m": round(context_dims["height_m"], 1),
            "wide_context_bbox_width_m": round(wide_dims["width_m"], 1),
            "wide_context_bbox_height_m": round(wide_dims["height_m"], 1),
        }

    if payload.lat is not None and payload.lng is not None and not payload.force_geocode:
        chip_size_m = payload.chip_size_m or 180
        bbox = make_square_bbox_from_point(payload.lat, payload.lng, chip_size_m=chip_size_m)
        context_bbox = make_square_bbox_from_point(payload.lat, payload.lng, chip_size_m=max(chip_size_m * 1.5, 240))
        wide_context_bbox = make_square_bbox_from_point(payload.lat, payload.lng, chip_size_m=max(chip_size_m * 2.4, 320))
        subzones = split_bbox_into_subzones(bbox, payload.lat)

        dims = bbox_width_height_m(bbox, payload.lat)
        context_dims = bbox_width_height_m(context_bbox, payload.lat)
        wide_dims = bbox_width_height_m(wide_context_bbox, payload.lat)

        return {
            "lat": payload.lat,
            "lng": payload.lng,
            "location_source": "user_supplied_coordinates",
            "matched_address": None,
            "polygon": None,
            "bbox": bbox,
            "context_bbox": context_bbox,
            "wide_context_bbox": wide_context_bbox,
            "subzones": subzones,
            "bbox_width_m": round(dims["width_m"], 1),
            "bbox_height_m": round(dims["height_m"], 1),
            "context_bbox_width_m": round(context_dims["width_m"], 1),
            "context_bbox_height_m": round(context_dims["height_m"], 1),
            "wide_context_bbox_width_m": round(wide_dims["width_m"], 1),
            "wide_context_bbox_height_m": round(wide_dims["height_m"], 1),
        }

    geocoded = geocode_address_mapbox(payload.address)
    if geocoded:
        chip_size_m = payload.chip_size_m or 180
        bbox = make_square_bbox_from_point(geocoded["lat"], geocoded["lng"], chip_size_m=chip_size_m)
        context_bbox = make_square_bbox_from_point(geocoded["lat"], geocoded["lng"], chip_size_m=max(chip_size_m * 1.5, 240))
        wide_context_bbox = make_square_bbox_from_point(geocoded["lat"], geocoded["lng"], chip_size_m=max(chip_size_m * 2.4, 320))
        subzones = split_bbox_into_subzones(bbox, geocoded["lat"])

        dims = bbox_width_height_m(bbox, geocoded["lat"])
        context_dims = bbox_width_height_m(context_bbox, geocoded["lat"])
        wide_dims = bbox_width_height_m(wide_context_bbox, geocoded["lat"])

        return {
            "lat": geocoded["lat"],
            "lng": geocoded["lng"],
            "location_source": "mapbox_geocoded_address",
            "matched_address": geocoded.get("address_matched"),
            "polygon": None,
            "bbox": bbox,
            "context_bbox": context_bbox,
            "wide_context_bbox": wide_context_bbox,
            "subzones": subzones,
            "bbox_width_m": round(dims["width_m"], 1),
            "bbox_height_m": round(dims["height_m"], 1),
            "context_bbox_width_m": round(context_dims["width_m"], 1),
            "context_bbox_height_m": round(context_dims["height_m"], 1),
            "wide_context_bbox_width_m": round(wide_dims["width_m"], 1),
            "wide_context_bbox_height_m": round(wide_dims["height_m"], 1),
        }

    raise HTTPException(status_code=400, detail="Could not resolve site location from polygon, coordinates, or address.")


def is_likely_qld(lat: float, lng: float) -> bool:
    return (-29.5 <= lat <= -9.0) and (137.0 <= lng <= 154.5)


def get_qld_historical_candidates(lat: float, lng: float) -> List[Dict[str, Any]]:
    url = f"{QLD_IMAGE_SERVER}/query"
    params = {
        "geometry": f"{lng},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "false",
        "outFields": ",".join([
            "objectid",
            "name",
            "year",
            "title",
            "capturestart",
            "captureend",
            "product_type",
            "res_value",
            "res_unit",
            "is_latest_public",
            "is_public",
            "category",
        ]),
        "orderByFields": "year ASC,res_value ASC",
        "f": "json",
    }

    data = safe_get(url, params)
    if not data:
        return []

    candidates = []
    for feature in data.get("features", []):
        attrs = feature.get("attributes", {}) or {}
        objectid = attrs.get("objectid")
        year = attrs.get("year")
        if not objectid or not year:
            continue

        candidates.append({
            "objectid": objectid,
            "name": attrs.get("name"),
            "year": year,
            "title": attrs.get("title"),
            "capture_start": epoch_ms_to_iso(attrs.get("capturestart")),
            "capture_end": epoch_ms_to_iso(attrs.get("captureend")),
            "product_type": attrs.get("product_type"),
            "res_value": attrs.get("res_value"),
            "res_unit": attrs.get("res_unit"),
            "is_latest_public": attrs.get("is_latest_public"),
            "is_public": attrs.get("is_public"),
            "category": attrs.get("category"),
        })
    return candidates


def scene_quality_score(scene: Dict[str, Any]) -> float:
    score = 1000.0
    if scene.get("product_type") in (3, 4, 5, 6, 7, 16):
        score -= 200
    if scene.get("is_public") == 1:
        score -= 50
    if scene.get("is_latest_public") == 1:
        score -= 25
    res_value = scene.get("res_value")
    if isinstance(res_value, (int, float)):
        score += float(res_value)
    return score


def dedupe_best_scene_per_year(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_year: Dict[int, Dict[str, Any]] = {}
    for scene in candidates:
        year = scene["year"]
        existing = by_year.get(year)
        if existing is None or scene_quality_score(scene) < scene_quality_score(existing):
            by_year[year] = scene
    return sorted(by_year.values(), key=lambda x: x["year"])


def pick_time_spread_scenes(candidates: List[Dict[str, Any]], max_scenes: int = 5) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    yearly = dedupe_best_scene_per_year(candidates)
    if len(yearly) <= max_scenes:
        return yearly

    idxs = sorted(set([
        0,
        len(yearly) // 4,
        len(yearly) // 2,
        (3 * len(yearly)) // 4,
        len(yearly) - 1,
    ]))
    return [yearly[i] for i in idxs][:max_scenes]


def classify_scene_budget_priority(analysis: Dict[str, Any]) -> Tuple[bool, bool]:
    obvious_current_pond_exists = False
    buried_secondary_risk_exists = False

    for f in safe_list(analysis.get("distinct_features")):
        ftype = safe_str(f.get("feature_type"), "other")
        conf = safe_str(f.get("confidence"), "low")
        notes = safe_str(f.get("notes"), "").lower()
        years = [int(y) for y in safe_list(f.get("detected_in_years")) if isinstance(y, (int, float))]

        if ftype == "pond" and conf in ("medium", "high"):
            obvious_current_pond_exists = True

        if ftype in ("former_pond", "probable_pond", "depression"):
            if any(y <= 2015 for y in years) or "historical water" in notes or "open water" in notes:
                buried_secondary_risk_exists = True

    return obvious_current_pond_exists, buried_secondary_risk_exists


def pick_followup_scenes(
    all_candidates: List[Dict[str, Any]],
    initial_selected: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    max_scenes: int = 8
) -> List[Dict[str, Any]]:
    if not all_candidates:
        return []

    yearly = dedupe_best_scene_per_year(all_candidates)
    all_years = [s["year"] for s in yearly]
    selected_years = [s["year"] for s in initial_selected]

    findings = ensure_findings_dict(analysis.get("historical_findings"))
    ponds = safe_dict(findings.get("former_ponds_dams"))
    fill = safe_dict(findings.get("fill_or_disturbance"))

    obvious_current_pond_exists, buried_secondary_risk_exists = classify_scene_budget_priority(analysis)

    suspicious = (
        ponds.get("status") in ("possible", "likely", "strong_evidence")
        or fill.get("status") in ("possible", "likely", "strong_evidence")
        or len(safe_list(analysis.get("distinct_features"))) >= 1
    )

    if not suspicious:
        return []

    candidate_years = set()
    for i in range(len(selected_years) - 1):
        y1 = selected_years[i]
        y2 = selected_years[i + 1]
        if y2 - y1 >= 3:
            for y in all_years:
                if y1 < y < y2:
                    candidate_years.add(y)

    for y in all_years:
        if y not in selected_years:
            candidate_years.add(y)

    followup = [s for s in yearly if s["year"] in candidate_years]
    followup = sorted(followup, key=lambda x: x["year"])

    if obvious_current_pond_exists and buried_secondary_risk_exists:
        followup = sorted(
            followup,
            key=lambda x: (
                0 if x["year"] <= 2015 else 1,
                x["year"]
            )
        )

    if len(followup) > max_scenes:
        if obvious_current_pond_exists and buried_secondary_risk_exists:
            early = [s for s in followup if s["year"] <= 2015]
            late = [s for s in followup if s["year"] > 2015]
            trimmed = early[:max_scenes]
            if len(trimmed) < max_scenes:
                trimmed.extend(late[: max_scenes - len(trimmed)])
            followup = trimmed
        else:
            idxs = sorted(set([
                0,
                len(followup) // 5,
                (2 * len(followup)) // 5,
                (3 * len(followup)) // 5,
                (4 * len(followup)) // 5,
                len(followup) - 1,
            ]))
            followup = [followup[i] for i in idxs][:max_scenes]

    return followup


def export_qld_historical_chip(scene: Dict[str, Any], bbox: Dict[str, float], size_px: int = 700) -> Optional[str]:
    objectid = scene.get("objectid")
    if not objectid:
        return None

    url = f"{QLD_IMAGE_SERVER}/exportImage"
    mosaic_rule = {
        "mosaicMethod": "esriMosaicLockRaster",
        "lockRasterIds": [objectid],
    }

    params = {
        "bbox": bbox_to_string(bbox),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{size_px},{size_px}",
        "format": "jpgpng",
        "interpolation": "RSP_BilinearInterpolation",
        "mosaicRule": json.dumps(mosaic_rule),
        "f": "json",
    }

    data = safe_get(url, params)
    if not data:
        return None
    return data.get("href")



def build_mapbox_bbox_image(
    bbox: Dict[str, float],
    polygon: Optional[List[List[float]]],
    label: str,
    image_type: str,
    source: str,
    year: str = "current",
    size_px: int = 700,
    padding_px: int = 60
) -> Optional[Dict[str, Any]]:
    if not MAPBOX_TOKEN:
        return None

    overlays = ""
    style = "satellite-streets-v12"

    if polygon:
        polygon = ensure_closed_polygon(polygon)
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "stroke": "#d7ac52",
                        "stroke-width": 3,
                        "stroke-opacity": 1,
                        "fill": "#d7ac52",
                        "fill-opacity": 0.10,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon],
                    },
                }
            ],
        }

        geojson_str = json.dumps(feature_collection, separators=(",", ":"))
        geojson_encoded = quote(geojson_str, safe="")
        overlays = f"geojson({geojson_encoded})/"

    url = (
        f"{MAPBOX_BASE_URL}/{style}/static/"
        f"{overlays}"
        f"[{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}]"
        f"/{size_px}x{size_px}"
        f"?padding={padding_px}&access_token={MAPBOX_TOKEN}"
    )

    return {
        "label": label,
        "type": image_type,
        "year": year,
        "capture_date": None,
        "source": source,
        "url": url,
        "bbox": bbox,
    }

def build_current_mapbox_images(
    lat: float,
    lng: float,
    bbox: Dict[str, float],
    context_bbox: Dict[str, float],
    wide_context_bbox: Dict[str, float],
    subzones: List[Dict[str, Any]],
    polygon: Optional[List[List[float]]] = None
) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []

    configs = [
        {"bbox": wide_context_bbox, "label": "current_wide_context", "image_type": "wide_context", "source": "mapbox_bbox_wide_context", "padding_px": 50},
        {"bbox": context_bbox, "label": "current_context", "image_type": "context", "source": "mapbox_bbox_context", "padding_px": 46},
        {"bbox": bbox, "label": "current_site", "image_type": "site", "source": "mapbox_bbox_site", "padding_px": 40},
    ]

    for cfg in configs:
        img = build_mapbox_bbox_image(
            bbox=cfg["bbox"],
            polygon=polygon,
            label=cfg["label"],
            image_type=cfg["image_type"],
            source=cfg["source"],
            size_px=700,
            padding_px=cfg["padding_px"],
        )
        if img:
            images.append(img)

    for zone in subzones[:6]:
        img = build_mapbox_bbox_image(
            bbox=zone["bbox"],
            polygon=polygon,
            label=f"current_{zone['label']}",
            image_type="subzone",
            source="mapbox_bbox_subzone",
            size_px=700,
            padding_px=30,
        )
        if img:
            images.append(img)

    if not images:
        images.append({
            "label": "current_site",
            "type": "site",
            "year": "current",
            "capture_date": None,
            "source": "mapbox_centroid_fallback",
            "url": f"{MAPBOX_BASE_URL}/satellite-streets-v12/static/{lng},{lat},18/700x700?access_token={MAPBOX_TOKEN}",
            "bbox": bbox,
        })

    return images



def build_historical_images_from_scenes(
    scenes: List[Dict[str, Any]],
    bbox: Dict[str, float],
    context_bbox: Optional[Dict[str, float]] = None,
    subzones: Optional[List[Dict[str, Any]]] = None,
    label_prefix: str = "historical_qld",
    include_context_for_edge_years: bool = True,
    include_subzones_for_edge_years: bool = True,
    prioritize_subzones: bool = False
) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []

    for idx, scene in enumerate(scenes, start=1):
        chip_url = export_qld_historical_chip(scene=scene, bbox=bbox, size_px=700)
        if chip_url:
            images.append({
                "label": f"{label_prefix}_{idx}",
                "type": "historical",
                "year": scene.get("year"),
                "capture_date": scene.get("capture_start") or scene.get("capture_end"),
                "source": "qld_public_imagery",
                "scene_objectid": scene.get("objectid"),
                "scene_title": scene.get("title"),
                "resolution_value": scene.get("res_value"),
                "res_unit": scene.get("res_unit"),
                "product_type": scene.get("product_type"),
                "url": chip_url,
                "bbox": bbox,
            })

    if include_context_for_edge_years and context_bbox and scenes:
        edge_scenes = [scenes[0]]
        if len(scenes) > 1:
            edge_scenes.append(scenes[-1])

        for idx, scene in enumerate(edge_scenes, start=1):
            chip_url = export_qld_historical_chip(scene=scene, bbox=context_bbox, size_px=700)
            if chip_url:
                images.append({
                    "label": f"{label_prefix}_context_{idx}",
                    "type": "historical_context",
                    "year": scene.get("year"),
                    "capture_date": scene.get("capture_start") or scene.get("capture_end"),
                    "source": "qld_public_imagery_context",
                    "scene_objectid": scene.get("objectid"),
                    "scene_title": scene.get("title"),
                    "resolution_value": scene.get("res_value"),
                    "res_unit": scene.get("res_unit"),
                    "product_type": scene.get("product_type"),
                    "url": chip_url,
                    "bbox": context_bbox,
                })

    if include_subzones_for_edge_years and subzones and scenes:
        target_scenes = [scenes[0]]
        if len(scenes) > 1:
            target_scenes.append(scenes[-1])

        if prioritize_subzones and len(scenes) > 2:
            mid = scenes[len(scenes) // 2]
            if mid not in target_scenes:
                target_scenes.append(mid)

        for scene_idx, scene in enumerate(target_scenes, start=1):
            for zone in subzones[:6]:
                chip_url = export_qld_historical_chip(scene=scene, bbox=zone["bbox"], size_px=700)
                if chip_url:
                    images.append({
                        "label": f"{label_prefix}_{zone['label']}_{scene_idx}",
                        "type": "historical_subzone",
                        "year": scene.get("year"),
                        "capture_date": scene.get("capture_start") or scene.get("capture_end"),
                        "source": "qld_public_imagery_subzone",
                        "scene_objectid": scene.get("objectid"),
                        "scene_title": scene.get("title"),
                        "resolution_value": scene.get("res_value"),
                        "res_unit": scene.get("res_unit"),
                        "product_type": scene.get("product_type"),
                        "url": chip_url,
                        "bbox": zone["bbox"],
                    })

    return images

def build_historical_qld_images(
    lat: float,
    lng: float,
    bbox: Dict[str, float],
    context_bbox: Dict[str, float],
    subzones: List[Dict[str, Any]],
    max_scenes: int = 5
) -> Dict[str, Any]:
    if not is_likely_qld(lat, lng):
        return {"images": [], "all_candidates": [], "selected_scenes": []}

    all_candidates = get_qld_historical_candidates(lat, lng)
    selected = pick_time_spread_scenes(all_candidates, max_scenes=max_scenes)
    images = build_historical_images_from_scenes(
        selected,
        bbox=bbox,
        context_bbox=context_bbox,
        subzones=subzones,
        label_prefix="historical_qld",
        include_context_for_edge_years=True,
        include_subzones_for_edge_years=True,
        prioritize_subzones=False,
    )

    return {
        "images": images,
        "all_candidates": all_candidates,
        "selected_scenes": selected,
    }


def filter_accessible_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_images = []
    for image in images:
        url = image.get("url")
        if not url:
            continue
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "image" in content_type or any(ext in url.lower() for ext in [".png", ".jpg", ".jpeg"]):
                valid_images.append(image)
        except Exception:
            continue
    return valid_images


def prioritize_ai_images(images: List[Dict[str, Any]], max_images: int = MAX_AI_IMAGES) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    used_labels = set()

    preferred_labels = [
        "current_wide_context",
        "current_context",
        "current_site",
        "current_subzone_nw",
        "current_subzone_ne",
        "current_subzone_sw",
        "current_subzone_se",
        "current_subzone_west",
        "current_subzone_east",
    ]

    for label in preferred_labels:
        for img in images:
            if img.get("label") == label and label not in used_labels:
                ordered.append(img)
                used_labels.add(label)

    historical_priority = sorted(
        [img for img in images if img.get("type") == "historical"],
        key=lambda x: x.get("year") if isinstance(x.get("year"), int) else 9999
    )
    historical_context = sorted(
        [img for img in images if img.get("type") == "historical_context"],
        key=lambda x: x.get("year") if isinstance(x.get("year"), int) else 9999
    )
    historical_subzone = sorted(
        [img for img in images if img.get("type") == "historical_subzone"],
        key=lambda x: (x.get("year") if isinstance(x.get("year"), int) else 9999, x.get("label", ""))
    )

    for collection in (historical_priority, historical_context, historical_subzone):
        for img in collection:
            label = img.get("label")
            if label not in used_labels:
                ordered.append(img)
                used_labels.add(label)

    return ordered[:max_images]


def build_image_manifest(images: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, image in enumerate(images, start=1):
        lines.append(
            f"{idx}. label={image.get('label')}; "
            f"type={image.get('type')}; "
            f"source={image.get('source')}; "
            f"year={image.get('year')}; "
            f"capture_date={image.get('capture_date')}"
        )
    return "\n".join(lines)


def simple_water_indicator(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    historical_years = sorted(
        [
            img["year"]
            for img in images
            if img.get("type") in ("historical", "historical_context", "historical_subzone")
            and isinstance(img.get("year"), int)
        ]
    )

    if not historical_years:
        return {
            "water_detected": False,
            "confidence": "low",
            "notes": "No historical imagery available, so former water features cannot be screened reliably.",
        }

    early_years = [y for y in historical_years if y <= 2005]
    late_years = [y for y in historical_years if y >= 2016]

    if early_years and late_years:
        return {
            "water_detected": True,
            "confidence": "medium",
            "notes": (
                f"Historical imagery spans early and recent years ({historical_years}). "
                "Screen carefully for former dams/ponds that may have existed in older imagery and changed over time."
            ),
        }

    if early_years:
        return {
            "water_detected": True,
            "confidence": "medium",
            "notes": (
                f"Older historical imagery is available ({historical_years}). "
                "Former dams/ponds may be visible in early imagery."
            ),
        }

    return {
        "water_detected": False,
        "confidence": "low",
        "notes": (
            f"Only limited mid/recent historical imagery is available ({historical_years}). "
            "Former water features may still be present, but early-condition screening is limited."
        ),
    }




def build_geotech_prompt(
    payload: SiteRequest,
    resolved: Dict[str, Any],
    image_manifest: str,
    water_signal: Dict[str, Any],
    mode: str = "initial",
    primary_features: Optional[List[Dict[str, Any]]] = None
) -> str:
    primary_text = ""
    if primary_features:
        rows = []
        for f in primary_features[:6]:
            rows.append(
                f"- type={f.get('feature_type', 'other')}, "
                f"relation={f.get('location_relation', 'on_site')}, "
                f"notes={f.get('notes', '')}"
            )
        primary_text = "\n".join(rows)

    prompt = f"""
You are reviewing north-aligned aerial imagery for geotechnical site-history screening.

IMPORTANT:
- All images are north-aligned. The top of each image is north, the bottom is south, the left is west, and the right is east.
- Your job is ONLY to identify candidate visual features per image.
- Keep outputs concise, professional, factual, and conservative.
- Do NOT assign final stable IDs like Pond A / Pond B / Pond C.
- Do NOT produce a polished final report narrative.
- Prefer clear visual evidence over speculation.
- Tight bounding boxes only around the visible feature footprint.
- Avoid long compass-heavy wording in notes. Prefer plain descriptions like "upper right", "central area", "road-facing side", or "site edge".
- This is a preliminary visual screening for obvious geotechnical risk indicators visible in imagery only.
- Do NOT assign an AS2870 site class from imagery.
- Identify visual indicators that may be relevant to AS2870 investigation planning, including abnormal moisture conditions.
- Any water setting can be relevant to abnormal moisture conditions under AS2870-style site assessment, but the TYPE of water setting must be classified correctly.

SITE:
- Address: {payload.address}
- Latitude: {resolved["lat"]}
- Longitude: {resolved["lng"]}
- Site bbox: {bbox_to_string(resolved["bbox"])}

IMAGE SET:
{image_manifest}

WATER SIGNAL:
- water_detected: {water_signal['water_detected']}
- water_confidence: {water_signal['confidence']}
- water_notes: {water_signal['notes']}

Return valid JSON only in this structure:
{{
  "summary": "1-3 short sentences only",
  "on_site_summary": "1 short sentence only",
  "adjacent_context_summary": "1 short sentence only",
  "candidates": [
    {{
      "feature_id": "feature_1",
      "feature_type": "pond_on_site | former_pond_or_dam | canal_edge_or_reclaimed_waterway | creek_or_drainage_line | beach_foreshore_or_coastal_edge | large_external_waterbody | possible_reclaimed_ground | fill_or_disturbance | existing_structure | former_structure | hardstand_or_slab | retaining_or_cut_fill | significant_tree_or_vegetation | uncertain_water_related_feature | water_candidate | disturbance_candidate | drainage_candidate | structure_candidate | earthworks_candidate | stockpile_candidate | other",
      "location_relation": "on_site | adjacent | off_site_context | uncertain",
      "confidence": "low | medium | high",
      "notes": "short visual description only",
      "evidence": ["historical water visible", "linear canal edge", "site changed from wet/low land to developed lot"],
      "detected_in_years": [1995, 2015, 2021],
      "primary_image_label": "historical_qld_1",
      "approximate_bbox_norm": [0.10, 0.20, 0.18, 0.22],
      "risk_priority": "primary | secondary | contextual"
    }}
  ],
  "feature_timeline": [],
  "historical_findings": {{
    "former_ponds_dams": {{"status": "none | possible | likely | strong_evidence", "confidence": "low | medium | high", "notes": "short note only"}},
    "vegetation_clearing": {{"status": "none | minor | moderate | major", "confidence": "low | medium | high", "notes": "short note only"}},
    "fill_or_disturbance": {{"status": "none | possible | likely | strong_evidence", "confidence": "low | medium | high", "notes": "short note only"}}
  }},
  "possible_risks": [],
  "recommended_investigation_focus": [],
  "recommended_follow_up": [],
  "visible_observations": [],
  "change_timeline": [],
  "screening_outcome": "1 short sentence only",
  "limitations": [],
  "confidence_overall": "low | medium | high"
}}

WATER SETTING CLASSIFICATION — CRITICAL:
You MUST return a candidate for any visible water body within the site, touching the site boundary, or immediately adjacent to the site in current_site/current_context/current_wide_context imagery.
Do not ignore adjacent water just because it is outside the lot. Adjacent water is still relevant to abnormal moisture and AS2870-style investigation planning.
For canal-front residential lots, the rear canal MUST be returned as canal_edge_or_reclaimed_waterway with location_relation=adjacent unless clearly inside the lot.
Before calling anything a pond, classify the water / wet feature setting as one of:
1. pond_on_site
   - isolated contained waterbody within the lot boundary
   - contained basin or depression
   - rounded, irregular, or enclosed shape
   - does not continue beyond the property or image frame as part of a connected water system
2. former_pond_or_dam
   - visible in historical imagery but no longer visible in current imagery
   - may show as dark depression, ring vegetation, tonal contrast, altered surface pattern, or infilled basin
   - likely infilled or modified over time
3. canal_edge_or_reclaimed_waterway
   - long, linear, engineered, or connected waterbody, commonly along a rear boundary or estate edge
   - continues beyond the site boundary or image frame
   - may have straight edges, revetments, retaining walls, pontoons, jetties, boat ramps, or canal lots
   - may indicate canal estate fill, dredged material, reclamation, groundwater influence, or variable founding conditions
4. creek_or_drainage_line
   - narrow linear natural or semi-natural drainage path
   - may be vegetated, sinuous, or connected to overland flow
   - may indicate alluvial, soft, or moisture-variable ground
5. beach_foreshore_or_coastal_edge
   - adjacent to beach, dune, foreshore, tidal flat, estuary, bay, or coastal sand environment
   - may indicate loose sand, groundwater influence, marine deposits, erosion, or acid sulfate soil context
6. large_external_waterbody
   - river, lake, bay, ocean, broad canal basin, or major external waterbody
   - treat as contextual unless the site appears reclaimed, low-lying, or directly modified
7. uncertain_water_related_feature
   - use where water influence is possible but the setting is unclear

STRICT WATER RULES:
- Do not classify a long, linear, connected waterbody as a pond.
- Do not classify a beach, foreshore, ocean edge, bay edge, estuary edge, or tidal flat as a pond.
- If a waterbody continues beyond the image or beyond the site boundary, prefer canal_edge_or_reclaimed_waterway, creek_or_drainage_line, beach_foreshore_or_coastal_edge, or large_external_waterbody over pond_on_site.
- Only classify as pond_on_site where the waterbody is isolated, contained, and located inside the lot boundary.
- If water is adjacent to the site but outside the lot, describe it as contextual unless there is visual evidence that the lot itself was reclaimed, filled, low-lying, or historically part of the waterbody.
- All water contexts may be relevant to abnormal moisture conditions, but do not overstate direct on-site risk unless the visual evidence is on-site.
- Adjacent canal / creek / foreshore / external water must still be returned as a contextual feature, not dropped from the candidate list.
- The correct outcome for a canal-front lot is usually: no isolated on-site pond, but adjacent canal / abnormal moisture / possible canal-edge fill context present.

CANAL / RECLAIMED WATERWAY INTERPRETATION:
For canal-front or reclaimed-waterway sites, look for:
- historical shoreline or canal alignment change
- former water or marshland replaced by residential lots
- straight canal edge or engineered water boundary
- fill extending from former canal / wetland / low-lying land
- lots created over former water-adjacent ground
- uniform estate development over reclaimed land
If present, use canal_edge_or_reclaimed_waterway or possible_reclaimed_ground rather than former_pond_or_dam unless there is a separate isolated pond/dam footprint.

BEACH / FORESHORE INTERPRETATION:
If the site is near a beach, dune, tidal flat, estuary, bay, ocean edge, or coastal foreshore, classify the setting separately from pond/wet depression.
Look for coastal sand/dune morphology, beach ridges, tidal margins, low-lying coastal land, historical shoreline movement, or reclaimed foreshore land.

FILL / DISTURBANCE INTERPRETATION:
Look for evidence of:
- former waterbody or low area becoming developed land
- earthworks, bulk earthworks, machine tracks, dozer tracks, scraped/stripped ground, regrading, filling, levelling, cut/fill platforms, retaining walls, batters, benching or terracing
- abrupt tonal changes between years
- bare ground, stockpiles, access tracks, construction disturbance
- building pads, slabs, hardstand, driveway construction
- former structures removed between imagery years
Classify as fill_or_disturbance, possible_reclaimed_ground, retaining_or_cut_fill, hardstand_or_slab, existing_structure, or former_structure as appropriate.

CUT / FILL INTERPRETATION RULE:
If earthworks, exposed soil, platform preparation, or disturbed ground is observed:
- Do not assume the presence of fill from exposed soil alone.
- Do not assume the site is entirely in cut from exposed soil alone.
- Treat general earthworks as cut/fill platform preparation or reworked ground unless clearer evidence is visible.
Only infer fill more strongly where there is clear visual evidence such as:
- the site appears raised relative to surrounding terrain
- a former depression, pond, waterway, or low area has been infilled
- historical imagery shows progressive infilling or reclamation
- retaining walls, batters, or edge geometry indicate placed material
- a platform extends beyond the natural slope alignment
If these indicators are not clearly present, describe the condition as earthworks / reworked ground with cut/fill unknown and fill presence unconfirmed.
If machine tracks, benching, terracing, stripped ground, or bulk earthworks are visible within the lot, return an earthworks_candidate or fill_or_disturbance candidate even if no building, slab, hardstand, or water feature is present.

STRUCTURE CLASSIFICATION:
Identify separately:
- existing_structure
- former_structure
- hardstand_or_slab
- driveway or building platform if clear
Always return an existing_structure candidate where a clear on-site dwelling, house, building roof, shed, or obvious structure is visible in current_site/current_context imagery.
Existing residential dwellings should be reported as a visible site observation even where they are not a geotechnical risk feature.
Do NOT treat an ordinary existing dwelling as fill_or_disturbance unless there is separate visible evidence of demolition, earthworks, regrading, slab preparation, hardstand construction, retaining, cut/fill, or disturbed ground.
Look for roofs, rectangular slabs, sheds, dwellings, hardstand, driveways, removed structures in historical comparison, or former building footprints.
Former structures, slabs, hardstand and driveways may indicate localised regrading, service trenches, compaction variation, demolition fill, or altered surface drainage.

AS2870-RELEVANT SCREENING FACTORS:
Where visible from imagery, consider:
- fill or uncontrolled fill indicators
- former ponds, dams, canals, creeks, wet areas, depressions, beaches, foreshore margins, or external water influence
- reclaimed land or canal-edge development
- abnormal moisture conditions or potential moisture variation
- trees or significant vegetation close to future footing zones
- cut/fill platforms, retaining walls, batters, slopes, or drainage concentration
- existing or former structures, hardstand, slabs, driveways, or service corridors
- low-lying, floodplain-like, coastal, foreshore, dune, or marine sand settings

GENERAL RULES:
- If multiple water-related features are visible in the SAME image, return each separately.
- Do not require a perfectly circular shape for former ponds/dams.
- Irregular, elongated, partially vegetated, or partially infilled former ponds still count if they are isolated basin/depression features.
- Features that appear in one historical image and disappear later are important and should still be returned.
- Do not force detections on every image.
- Only return non-water geotechnical risk indicators where they are visually obvious.
- If a former pond or historical feature is only clear historically, bbox it on the historical image where it is clearest.
- Keep notes short. No more than 25 words each.
"""

    if mode == "remainder_rescan":
        prompt += f"""

REMAINDER RESCAN:
- focus on secondary former ponds / buried ponds / subtle circular basins
- also check whether any water-related feature is better classified as canal, drainage, foreshore, large external waterbody, or reclaimed ground
- aggressively look for a second pond only if it is a separate isolated basin/depression, not a canal/foreshore edge
- return faint, messy, irregular, or partly vegetated former pond footprints if visible
- current known higher priority features:
{primary_text if primary_text else "- none provided"}
"""
    elif mode == "hunter":
        prompt += f"""

HUNTER MODE:
- This is an aggressive low-confidence risk scan.
- Return faint, irregular, partially infilled, vegetated-ring, tonal, oval, circular, elongated, disconnected, or depression-like features that may indicate a former pond / buried pond / old dam footprint.
- Do NOT turn canals, foreshore edges, beaches, creeks, or broad external waterbodies into ponds.
- If two true isolated water/depression features are visible in the same image, return both separately.
- Prioritise features that appear in one historical image but are weak or absent in later imagery.
- Do not require a perfect circular shape.
- Prefer new or previously unreported on-site features.
- Low confidence candidates are acceptable.
- Return up to 4 candidate water features even if subtle.
- current known higher priority features:
{primary_text if primary_text else "- none provided"}
"""
    return prompt

def default_analysis_payload(raw_output: str = "") -> Dict[str, Any]:
    return {
        "summary": "AI output could not be parsed cleanly.",
        "on_site_summary": "Manual review required due to AI parsing failure.",
        "adjacent_context_summary": "No reliable adjacent/off-site context summary available.",
        "visible_observations": [],
        "change_timeline": [],
        "distinct_features": [],
        "feature_timeline": [],
        "historical_findings": ensure_findings_dict(None),
        "possible_risks": [],
        "limitations": ["AI output parsing failed. Manual review required."],
        "screening_outcome": "Unknown",
        "recommended_follow_up": [
            "Detailed geotechnical investigation is strongly recommended based on identified site history risks."
        ],
        "recommended_investigation_focus": [],
        "confidence_overall": "low",
    }



def normalize_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    bbox = clamp_norm_bbox(safe_list(feature.get("approximate_bbox_norm")))
    primary_label = safe_str(feature.get("primary_image_label"), "current_wide_context")
    return {
        "feature_id": safe_str(feature.get("feature_id"), "Unnamed"),
        "feature_type": safe_str(feature.get("feature_type"), "other"),
        "location_relation": safe_str(feature.get("location_relation"), "on_site"),
        "confidence": safe_str(feature.get("confidence"), "low"),
        "notes": safe_str(feature.get("notes"), ""),
        "evidence": safe_list(feature.get("evidence")),
        "detected_in_years": [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))],
        "primary_image_label": primary_label,
        "detected_on_image": primary_label,
        "approximate_bbox_norm": bbox,
        "risk_priority": safe_str(feature.get("risk_priority"), "secondary"),
        "persistence_score": int(feature.get("persistence_score", 0) or 0),
    }

def build_features_from_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    built: List[Dict[str, Any]] = []
    for c in candidates:
        feature = dict(c)
        raw_type = safe_str(feature.get("feature_type"), "other").lower()
        notes = safe_str(feature.get("notes"), "")
        evidence = [str(x) for x in safe_list(feature.get("evidence"))]
        text_blob = (notes + " " + " ".join(evidence)).lower()

        canal_terms = ["canal", "estate canal", "tidal canal", "linear canal", "waterway", "revetment", "pontoon", "jetty", "boat ramp"]
        creek_terms = ["creek", "drainage", "drain", "channel", "flow path", "overland flow"]
        beach_terms = ["beach", "foreshore", "dune", "tidal", "estuary", "ocean", "shoreline", "coastal", "bay"]
        reclaimed_terms = ["reclaimed", "reclamation", "dredged", "canal fill", "former canal", "filled canal", "marsh", "low-lying"]

        if raw_type in ("pond_on_site", "pond"):
            feature["feature_type"] = "pond"

        elif raw_type in ("former_pond_or_dam", "former_pond"):
            feature["feature_type"] = "former_pond"

        elif raw_type in ("water_candidate", "probable_pond"):
            if any(k in text_blob for k in canal_terms):
                feature["feature_type"] = "canal"
            elif any(k in text_blob for k in beach_terms):
                feature["feature_type"] = "beach_foreshore_or_coastal_edge"
            elif any(k in text_blob for k in creek_terms):
                feature["feature_type"] = "drainage_feature"
            elif any(k in text_blob for k in reclaimed_terms):
                feature["feature_type"] = "possible_reclaimed_ground"
            else:
                feature["feature_type"] = "probable_pond"

        elif raw_type in ("canal_edge_or_reclaimed_waterway", "canal"):
            feature["feature_type"] = "canal"
            feature["evidence"] = merge_string_lists(evidence + ["canal / reclaimed waterway context"])
            if "canal" not in text_blob and "waterway" not in text_blob:
                feature["notes"] = (notes + " canal / reclaimed waterway context").strip()

        elif raw_type in ("creek_or_drainage_line", "drainage_candidate", "drainage_feature"):
            if any(k in text_blob for k in canal_terms):
                feature["feature_type"] = "canal"
            else:
                feature["feature_type"] = "drainage_feature"
                if "creek" not in text_blob and "drain" not in text_blob and "channel" not in text_blob:
                    feature["notes"] = (notes + " drainage/creek-like linear water feature").strip()

        elif raw_type in ("beach_foreshore_or_coastal_edge",):
            feature["feature_type"] = "beach_foreshore_or_coastal_edge"
            feature["evidence"] = merge_string_lists(evidence + ["coastal / foreshore water context"])

        elif raw_type in ("large_external_waterbody",):
            feature["feature_type"] = "large_external_waterbody"

        elif raw_type in ("uncertain_water_related_feature",):
            feature["feature_type"] = "uncertain_water_related_feature"

        elif raw_type in ("possible_reclaimed_ground",):
            feature["feature_type"] = "possible_reclaimed_ground"
            feature["evidence"] = merge_string_lists(evidence + ["possible reclaimed / placed fill ground"])

        elif raw_type in ("structure_candidate", "existing_structure", "former_structure", "hardstand_or_slab"):
            historical_hint = raw_type == "former_structure" or any(
                k in text_blob for k in [
                    "former structure", "former dwelling", "removed structure", "demolished",
                    "old slab", "historical slab", "historical dwelling", "prior building",
                    "removed building", "former building"
                ]
            )
            hardstand_hint = raw_type == "hardstand_or_slab" or any(k in text_blob for k in ["hardstand", "slab", "driveway", "pavement"])
            feature["feature_type"] = "former_structure" if historical_hint else "hardstand_or_slab" if hardstand_hint else "existing_structure"
            if not any(k in text_blob for k in ["building", "dwelling", "house", "shed", "slab", "hardstand", "driveway", "structure", "roof"]):
                prefix = "Existing building / hardstand footprint visible" if not historical_hint else "Former building / hardstand footprint visible in historical imagery"
                feature["notes"] = f"{prefix}. {notes}".strip().strip(".")
            extra_evidence = [
                "building / hardstand footprint visible" if not historical_hint else "former building footprint visible",
                "possible prior localised ground modification",
            ]
            feature["evidence"] = merge_string_lists(evidence + extra_evidence)

        elif raw_type in ("fill_or_disturbance", "disturbance_candidate", "earthworks_candidate", "stockpile_candidate", "disturbance", "fill_area"):
            feature["feature_type"] = "disturbance"

        elif raw_type in ("retaining_or_cut_fill",):
            feature["feature_type"] = "retaining_or_cut_fill"

        elif raw_type in ("significant_tree_or_vegetation",):
            feature["feature_type"] = "significant_tree_or_vegetation"

        built.append(normalize_feature(feature))
    return built

def upgrade_geotech_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    upgraded: List[Dict[str, Any]] = []

    for f in features:
        feature = dict(f)
        notes = safe_str(feature.get("notes"), "").lower()
        evidence_items = [str(x) for x in safe_list(feature.get("evidence"))]
        evidence_text = " ".join(evidence_items).lower()
        text = notes + " " + evidence_text
        years = [int(y) for y in safe_list(feature.get("detected_in_years")) if isinstance(y, (int, float))]
        bbox = safe_list(feature.get("approximate_bbox_norm"))
        raw_type = safe_str(feature.get("feature_type"), "other").lower()

        contextual_types = (
            "canal", "drainage_feature", "beach_foreshore_or_coastal_edge",
            "large_external_waterbody", "uncertain_water_related_feature",
            "possible_reclaimed_ground", "existing_structure", "former_structure",
            "hardstand_or_slab", "retaining_or_cut_fill", "significant_tree_or_vegetation"
        )
        if raw_type in contextual_types:
            feature["feature_type"] = raw_type
            if raw_type in ("canal", "drainage_feature", "beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"):
                feature["risk_priority"] = "contextual" if safe_str(feature.get("location_relation"), "") != "on_site" else "secondary"
            elif raw_type == "possible_reclaimed_ground":
                feature["risk_priority"] = "secondary" if safe_str(feature.get("location_relation"), "") == "on_site" else "contextual"
            else:
                feature["risk_priority"] = "secondary"
            feature["approximate_bbox_norm"] = clamp_norm_bbox(bbox)
            upgraded.append(feature)
            continue

        canal_signal = any(k in text for k in ["canal", "estate canal", "tidal canal", "linear canal", "waterway", "revetment", "pontoon", "jetty"])
        creek_signal = any(k in text for k in ["creek", "drainage", "drain", "channel", "flow path"])
        beach_signal = any(k in text for k in ["beach", "foreshore", "dune", "tidal", "estuary", "ocean", "shoreline", "coastal", "bay"])
        reclaimed_signal = any(k in text for k in ["reclaimed", "reclamation", "dredged", "canal fill", "former canal", "filled canal", "marsh", "low-lying"])

        historical_visible = (
            any(y <= 2015 for y in years)
            or "historical water" in evidence_text
            or "older imagery" in evidence_text
            or "visible in 1995" in notes
            or "historic imagery" in notes
        )
        current_visible = likely_current_visible(feature)

        circularish = any(k in notes or k in evidence_text for k in [
            "circular", "semi-circular", "round", "rounded", "ring", "basin", "depression", "pond footprint"
        ])
        wet_signature = any(k in notes or k in evidence_text for k in [
            "water", "wet", "moist", "vegetated edge", "vegetated ring", "wet hollow",
            "moisture signature", "open water", "standing water", "waterbody", "water body"
        ])
        disturbance_signature = any(k in notes or k in evidence_text for k in [
            "graded", "earthworks", "cut", "fill edge", "track scar", "stockpile", "material pile",
            "disturbance", "disturbed ground", "cleared soil", "construction materials",
            "changed land cover", "surface disturbance", "bare soil", "earth scarring", "compacted",
            "retaining", "retaining wall", "cut platform", "benched", "benching", "levelled pad",
            "batter", "excavation", "machine tracks", "dozer tracks", "track marks", "tracked ground",
            "bulk earthworks", "platform preparation", "earthmoving", "scraped ground", "stripped ground",
            "bench", "terraced", "terracing", "dwelling", "house pad", "slab", "hardstand", "shed", "fill", "reworked"
        ])
        structure_adjacent_signature = any(k in notes or k in evidence_text for k in [
            "near buildings", "near structure", "around structures", "road access", "access road",
            "proximity to structures", "material piles", "existing dwelling", "existing structure",
            "shed", "slab", "hardstand", "house pad", "building footprint", "roof"
        ])

        if canal_signal:
            feature["feature_type"] = "canal"
        elif beach_signal:
            feature["feature_type"] = "beach_foreshore_or_coastal_edge"
        elif creek_signal:
            feature["feature_type"] = "drainage_feature"
        elif reclaimed_signal and not circularish:
            feature["feature_type"] = "possible_reclaimed_ground"
        elif disturbance_signature and not wet_signature:
            feature["feature_type"] = "disturbance"
        elif structure_adjacent_signature and raw_type in ("probable_pond", "other", "disturbance") and not wet_signature:
            feature["feature_type"] = "disturbance"
        elif current_visible and wet_signature and not disturbance_signature:
            feature["feature_type"] = "pond"
        elif historical_visible and not current_visible and (wet_signature or circularish) and not disturbance_signature:
            feature["feature_type"] = "former_pond"
        elif historical_visible and wet_signature and circularish and not disturbance_signature:
            feature["feature_type"] = "probable_pond"
        elif raw_type in ("fill_area", "disturbance"):
            feature["feature_type"] = raw_type
        else:
            feature["feature_type"] = "other"

        if feature["feature_type"] == "probable_pond":
            if not wet_signature:
                feature["feature_type"] = "disturbance" if disturbance_signature or structure_adjacent_signature else "other"
            elif feature.get("confidence") == "low" and (disturbance_signature or structure_adjacent_signature):
                feature["feature_type"] = "disturbance"

        if feature["feature_type"] == "pond":
            feature["risk_priority"] = "secondary"
        elif feature["feature_type"] == "former_pond":
            feature["risk_priority"] = "primary"
        elif feature["feature_type"] in ("probable_pond", "depression", "fill_area", "disturbance", "retaining_or_cut_fill", "possible_reclaimed_ground", "existing_structure", "former_structure", "hardstand_or_slab"):
            feature["risk_priority"] = "secondary"
        elif feature["feature_type"] in ("canal", "drainage_feature", "beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"):
            feature["risk_priority"] = "contextual" if safe_str(feature.get("location_relation"), "") != "on_site" else "secondary"
        elif safe_str(feature.get("location_relation"), "") in ("adjacent", "off_site_context"):
            feature["risk_priority"] = "contextual"
        else:
            feature["risk_priority"] = safe_str(feature.get("risk_priority"), "secondary")

        if len(years) >= 2 and feature["confidence"] == "low":
            feature["confidence"] = "medium"

        if feature["feature_type"] not in ("pond", "former_pond"):
            fid = safe_str(feature.get("feature_id"), "")
            if fid.lower().startswith(("pond ", "former pond", "active pond", "probable pond", "water_feature")):
                feature["feature_id"] = ""

        feature["approximate_bbox_norm"] = clamp_norm_bbox(bbox)
        upgraded.append(feature)

    return upgraded

def deduplicate_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Lightweight pass before clustering. Final identity resolution happens later.
    deduped: List[Dict[str, Any]] = []
    for feature in features:
        matched = False
        for existing in deduped:
            if should_merge_features(existing, feature):
                merge_feature_pair(existing, feature)
                matched = True
                break
        if not matched:
            deduped.append(dict(feature))
    return deduped


def assign_stable_feature_ids(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    on_site_water = [
        f for f in features
        if safe_str(f.get("feature_type"), "") in ("pond", "former_pond", "probable_pond")
        and safe_str(f.get("location_relation"), "") == "on_site"
    ]
    adjacent_water = [
        f for f in features
        if safe_str(f.get("feature_type"), "") in (
            "pond", "former_pond", "probable_pond", "drainage_feature", "canal",
            "beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"
        )
        and safe_str(f.get("location_relation"), "") != "on_site"
    ]
    structures = [
        f for f in features
        if safe_str(f.get("feature_type"), "") in ("existing_structure", "former_structure", "hardstand_or_slab")
    ]

    on_site_sorted = sorted(on_site_water, key=feature_sort_key_for_ids)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, feature in enumerate(on_site_sorted):
        feature["feature_id"] = f"Pond {letters[idx] if idx < len(letters) else idx + 1}"

    adj_count = 0
    canal_count = 0
    creek_count = 0
    coastal_count = 0
    large_water_count = 0
    for feature in sorted(adjacent_water, key=feature_sort_key_for_ids):
        ftype = safe_str(feature.get("feature_type"), "")
        notes = (safe_str(feature.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(feature.get("evidence"))])).lower()
        if ftype == "canal" or "canal" in notes:
            canal_count += 1
            feature["feature_id"] = "Adjacent Canal" if canal_count == 1 else f"Adjacent Canal {canal_count}"
        elif ftype == "beach_foreshore_or_coastal_edge" or any(k in notes for k in ["beach", "foreshore", "coastal", "shoreline", "tidal", "estuary"]):
            coastal_count += 1
            feature["feature_id"] = "Foreshore / Coastal Edge" if coastal_count == 1 else f"Foreshore / Coastal Edge {coastal_count}"
        elif ftype == "large_external_waterbody":
            large_water_count += 1
            feature["feature_id"] = "External Waterbody" if large_water_count == 1 else f"External Waterbody {large_water_count}"
        elif "creek" in notes or "drainage" in notes or ftype == "drainage_feature":
            creek_count += 1
            feature["feature_id"] = "Adjacent Creek / Drainage" if creek_count == 1 else f"Adjacent Creek / Drainage {creek_count}"
        else:
            adj_count += 1
            feature["feature_id"] = f"Adjacent Water Feature {adj_count}"

    existing_count = 0
    former_count = 0
    hardstand_count = 0
    for feature in sorted(structures, key=feature_sort_key_for_ids):
        ftype = safe_str(feature.get("feature_type"), "")
        if ftype == "former_structure":
            former_count += 1
            feature["feature_id"] = "Former Structure" if former_count == 1 else f"Former Structure {former_count}"
        elif ftype == "hardstand_or_slab":
            hardstand_count += 1
            feature["feature_id"] = "Hardstand / Slab" if hardstand_count == 1 else f"Hardstand / Slab {hardstand_count}"
        else:
            existing_count += 1
            feature["feature_id"] = "Existing Structure" if existing_count == 1 else f"Existing Structure {existing_count}"

    for feature in features:
        if safe_str(feature.get("feature_id"), "").strip():
            continue
        ftype = safe_str(feature.get("feature_type"), "feature")
        if ftype == "fill_area":
            feature["feature_id"] = "Fill Area"
        elif ftype == "disturbance":
            feature["feature_id"] = "Disturbance Area"
        elif ftype == "possible_reclaimed_ground":
            feature["feature_id"] = "Possible Reclaimed Ground"
        elif ftype == "retaining_or_cut_fill":
            feature["feature_id"] = "Retaining / Cut-Fill"
        elif ftype == "drainage_feature":
            feature["feature_id"] = "Drainage Feature"
        elif ftype == "canal":
            feature["feature_id"] = "Canal"
        elif ftype == "beach_foreshore_or_coastal_edge":
            feature["feature_id"] = "Foreshore / Coastal Edge"
        elif ftype == "large_external_waterbody":
            feature["feature_id"] = "External Waterbody"
        elif ftype == "uncertain_water_related_feature":
            feature["feature_id"] = "Uncertain Water Feature"
        elif ftype == "existing_structure":
            feature["feature_id"] = "Existing Structure"
        elif ftype == "former_structure":
            feature["feature_id"] = "Former Structure"
        elif ftype == "hardstand_or_slab":
            feature["feature_id"] = "Hardstand / Slab"
        elif ftype == "significant_tree_or_vegetation":
            feature["feature_id"] = "Significant Vegetation"
        else:
            feature["feature_id"] = ftype.replace("_", " ").title()

    return features

def finalize_feature_geometry_and_ids(
    features: List[Dict[str, Any]],
    resolved: Dict[str, Any],
    image_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    polygon = resolved.get("polygon")
    processed: List[Dict[str, Any]] = []

    for feature in features:
        bbox = safe_list(feature.get("approximate_bbox_norm"))
        primary_label = safe_str(feature.get("primary_image_label"), "current_wide_context")
        image_meta = safe_dict(image_lookup.get(primary_label))
        image_bbox = safe_dict(image_meta.get("bbox"))
        geo_bbox = norm_bbox_to_geo_bbox(bbox, image_bbox) if bbox and image_bbox else None

        feature["source_image_labels"] = merge_string_lists([primary_label] + safe_list(feature.get("source_image_labels")))
        feature["detected_on_image"] = safe_str(feature.get("detected_on_image"), primary_label)

        if geo_bbox:
            feature["geo_bbox"] = geo_bbox
            if polygon:
                feature["location_relation"] = resolve_feature_relation(
                    geo_bbox,
                    polygon,
                    feature_type=safe_str(feature.get("feature_type"), ""),
                    notes=safe_str(feature.get("notes"), ""),
                )
            else:
                feature["location_relation"] = safe_str(feature.get("location_relation"), "on_site")

        processed.append(feature)

    processed = merge_feature_sets_geometry_first([], processed)
    processed = [classify_cluster_feature(f, image_lookup) for f in processed]

    # Keep only meaningful final feature types for report output.
    processed = [
        f for f in processed
        if safe_str(f.get("feature_type"), "") in (
            "pond", "former_pond", "probable_pond", "disturbance", "fill_area",
            "drainage_feature", "canal", "beach_foreshore_or_coastal_edge",
            "large_external_waterbody", "uncertain_water_related_feature", "possible_reclaimed_ground",
            "existing_structure", "former_structure", "hardstand_or_slab", "retaining_or_cut_fill",
            "significant_tree_or_vegetation"
        )
    ]
    # remove duplicate disturbance boxes when they substantially overlap
    tmp=[]
    for f in processed:
        if safe_str(f.get("feature_type"),"") != "disturbance":
            tmp.append(f)
            continue
        fg = safe_dict(f.get("geo_bbox"))
        dup=False
        for e in tmp:
            if safe_str(e.get("feature_type"),"") == "disturbance" and overlap_ratio(fg, safe_dict(e.get("geo_bbox"))) >= 0.7:
                dup=True
                break
        if not dup:
            tmp.append(f)
    processed = tmp
    processed = deduplicate_features(processed)
    processed = collapse_near_identical_features(processed)
    processed = cluster_features_by_identity(processed)
    processed = dedupe_final_anchored_features(processed)

    processed = assign_stable_feature_ids(processed)

    for feature in processed:
        ftype = safe_str(feature.get("feature_type"), "other")
        relation = safe_str(feature.get("location_relation"), "on_site")
        if relation != "on_site":
            feature["risk_priority"] = "contextual"
        elif ftype == "former_pond":
            feature["risk_priority"] = "primary"
        elif ftype in ("pond", "probable_pond", "disturbance", "fill_area", "existing_structure", "former_structure", "hardstand_or_slab", "possible_reclaimed_ground", "retaining_or_cut_fill", "significant_tree_or_vegetation"):
            feature["risk_priority"] = "secondary"
        elif ftype in ("drainage_feature", "canal", "beach_foreshore_or_coastal_edge", "large_external_waterbody", "uncertain_water_related_feature"):
            feature["risk_priority"] = "contextual" if relation != "on_site" else "secondary"

    return processed


def parse_analysis_response(
    raw_output: str,
    resolved: Optional[Dict[str, Any]] = None,
    image_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cleaned_output = clean_json_text(raw_output)
    try:
        data = json.loads(cleaned_output)
    except json.JSONDecodeError:
        return default_analysis_payload(raw_output)

    if not isinstance(data, dict):
        return default_analysis_payload(raw_output)

    data.setdefault("summary", "")
    data.setdefault("on_site_summary", "")
    data.setdefault("adjacent_context_summary", "")
    data.setdefault("visible_observations", [])
    data.setdefault("change_timeline", [])
    data.setdefault("distinct_features", [])
    data.setdefault("feature_timeline", [])
    data.setdefault("historical_findings", {})
    data.setdefault("possible_risks", [])
    data.setdefault("recommended_investigation_focus", [])
    data.setdefault("recommended_follow_up", [])
    data.setdefault("screening_outcome", "")
    data.setdefault("limitations", [])
    data.setdefault("confidence_overall", "low")

    candidates = safe_list(data.get("candidates"))
    if not candidates:
        candidates = safe_list(data.get("distinct_features"))
    features = [normalize_feature(f) for f in candidates if isinstance(f, dict)]
    features = build_features_from_candidates(features)

    if resolved and image_lookup:
        features = finalize_feature_geometry_and_ids(features, resolved=resolved, image_lookup=image_lookup)

    for f in features:
        f["feature_id"] = sanitize_ai_feature_id(
            safe_str(f.get("feature_id"), ""),
            safe_str(f.get("feature_type"), "other"),
        )

    data["distinct_features"] = features

    on_site_pond_count = sum(
        1
        for f in features
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") in ("pond", "former_pond", "probable_pond")
    )
    if on_site_pond_count >= 2:
        extra = " Multiple pond features are present on site, including historical and/or buried pond signatures."
        if extra.strip() not in safe_str(data.get("on_site_summary"), ""):
            data["on_site_summary"] = (safe_str(data.get("on_site_summary"), "").rstrip() + extra).strip()
        summary_extra = " Historical imagery indicates multiple on-site pond features, including secondary or buried pond signatures."
        if summary_extra.strip() not in safe_str(data.get("summary"), ""):
            data["summary"] = (safe_str(data.get("summary"), "").rstrip() + summary_extra).strip()
    data["historical_findings"] = ensure_findings_dict(data.get("historical_findings"))
    data["feature_timeline"] = normalize_feature_timeline_items(data.get("feature_timeline", []))
    data["change_timeline"] = normalize_change_timeline_items(data.get("change_timeline", []))

    data["recommended_investigation_focus"] = []
    data["recommended_follow_up"] = [
        "Detailed geotechnical investigation is strongly recommended based on identified site history risks."
    ]

    cleaned_risks = []
    for feature in features:
        if safe_str(feature.get("feature_type"), "") == "former_pond" and safe_str(feature.get("location_relation"), "") == "on_site":
            cleaned_risks.append("Former pond signatures may indicate fill, uncontrolled fill, deep fill, soft ground, or moisture-sensitive soils.")
        elif safe_str(feature.get("feature_type"), "") == "fill_area" and safe_str(feature.get("location_relation"), "") == "on_site":
            cleaned_risks.append("Earthworks or reworked ground may indicate cut/fill platform preparation and variable near-surface conditions; fill presence is unconfirmed from imagery alone.")
        elif safe_str(feature.get("feature_type"), "") == "pond" and safe_str(feature.get("location_relation"), "") == "on_site":
            cleaned_risks.append("Active pond areas may indicate local wetness influence and moisture variability in surrounding ground.")
        elif safe_str(feature.get("feature_type"), "") == "possible_reclaimed_ground" and safe_str(feature.get("location_relation"), "") == "on_site":
            cleaned_risks.append("Possible reclaimed or canal-edge fill may indicate variable founding conditions and abnormal moisture influence.")
        elif safe_str(feature.get("feature_type"), "") in ("canal", "drainage_feature", "beach_foreshore_or_coastal_edge", "large_external_waterbody"):
            cleaned_risks.append("Adjacent water settings may be relevant to abnormal moisture conditions and AS2870 investigation planning.")

    if cleaned_risks:
        data["possible_risks"] = list(dict.fromkeys(cleaned_risks))
    elif data.get("possible_risks"):
        data["possible_risks"] = list(dict.fromkeys([str(x) for x in safe_list(data.get("possible_risks"))]))[:4]

    return data

def call_ai_screening(
    payload: SiteRequest,
    resolved: Dict[str, Any],
    images: List[Dict[str, Any]],
    water_signal: Dict[str, Any],
    mode: str = "initial",
    primary_features: Optional[List[Dict[str, Any]]] = None,
    max_images: int = MAX_AI_IMAGES,
) -> Dict[str, Any]:
    usable_images = filter_accessible_images(images)
    usable_images = prioritize_ai_images(usable_images, max_images=max_images)

    if not usable_images:
        raise HTTPException(status_code=500, detail="No valid imagery URLs were accessible for AI screening.")

    image_manifest = build_image_manifest(usable_images)
    prompt = build_geotech_prompt(
        payload=payload,
        resolved=resolved,
        image_manifest=image_manifest,
        water_signal=water_signal,
        mode=mode,
        primary_features=primary_features,
    )

    user_content = [{"type": "input_text", "text": prompt}]
    for image in usable_images:
        user_content.append({"type": "input_image", "image_url": image["url"]})

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI screening failed: {str(exc)}") from exc

    image_lookup = {img.get("label"): img for img in images}
    return parse_analysis_response(response.output_text, resolved=resolved, image_lookup=image_lookup)

def get_on_site_water_features(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = safe_list(analysis.get("distinct_features"))
    return [
        f for f in features
        if safe_str(f.get("location_relation"), "on_site") == "on_site"
        and safe_str(f.get("feature_type"), "other") in ("pond", "former_pond", "probable_pond", "depression", "possible_reclaimed_ground")
    ]

def get_feature_focus_labels(features: List[Dict[str, Any]]) -> List[str]:
    labels = []
    for f in safe_list(features):
        for lbl in [safe_str(f.get("detected_on_image"), ""), safe_str(f.get("primary_image_label"), "")]:
            if lbl and lbl not in labels:
                labels.append(lbl)
    return labels


def pick_hunter_images(images: List[Dict[str, Any]], primary_features: List[Dict[str, Any]], max_images: int = 12) -> List[Dict[str, Any]]:
    focus_labels = set(get_feature_focus_labels(primary_features))
    selected: List[Dict[str, Any]] = []
    used = set()

    def add(img: Dict[str, Any]):
        lbl = img.get("label")
        if lbl and lbl not in used:
            selected.append(img)
            used.add(lbl)

    # Always include the oldest and latest full historical views.
    # Secondary / buried ponds are often most obvious in these wide historical frames.
    full_historical = sorted(
        [img for img in images if img.get("type") == "historical"],
        key=lambda x: (x.get("year") if isinstance(x.get("year"), int) else 9999, x.get("label", "")),
    )
    historical_context = sorted(
        [img for img in images if img.get("type") == "historical_context"],
        key=lambda x: (x.get("year") if isinstance(x.get("year"), int) else 9999, x.get("label", "")),
    )
    if full_historical:
        add(full_historical[0])
        if len(full_historical) > 1:
            add(full_historical[-1])
    if historical_context:
        add(historical_context[0])

    preferred_types = [
        "historical_subzone",
        "historical_context",
        "historical",
        "subzone",
        "site",
        "context",
    ]

    # Prioritise non-primary views so the model hunts for new features.
    for ptype in preferred_types:
        pool = [img for img in images if img.get("type") == ptype and img.get("label") not in focus_labels]
        if ptype.startswith("historical"):
            pool = sorted(pool, key=lambda x: (x.get("year") if isinstance(x.get("year"), int) else 9999, x.get("label", "")))
        for img in pool:
            add(img)
            if len(selected) >= max_images:
                return selected[:max_images]

    # Anchor with a couple of current views.
    for lbl in ["current_site", "current_context", "current_wide_context"]:
        for img in images:
            if img.get("label") == lbl:
                add(img)
                break

    # Finally include primary labels only as last resort reference.
    for img in images:
        if img.get("label") in focus_labels:
            add(img)
            if len(selected) >= max_images:
                break

    return selected[:max_images]


def hunter_keep_feature(feature: Dict[str, Any], primary_features: List[Dict[str, Any]]) -> bool:
    ftype = safe_str(feature.get("feature_type"), "other")
    relation = safe_str(feature.get("location_relation"), "uncertain")
    conf = safe_str(feature.get("confidence"), "low")
    text = (safe_str(feature.get("notes"), "") + " " + " ".join([str(x) for x in safe_list(feature.get("evidence"))])).lower()
    years = feature_year_set(feature)

    # Let later geometry decide on-site vs adjacent, but drop explicit off-site context.
    if relation == "off_site_context":
        return False
    if ftype not in ("pond", "former_pond", "probable_pond", "depression"):
        return False
    if any(k in text for k in ["canal", "foreshore", "beach", "ocean", "bay", "estuary", "shoreline", "tidal flat", "linear waterway"]):
        return False
    if not any(k in text for k in ["pond", "water", "basin", "depression", "ring", "wet", "circular", "oval", "dark", "elongated", "irregular", "semi-circular"]):
        return False
    if conf not in ("low", "medium", "high"):
        return False

    fg = safe_dict(feature.get("geo_bbox"))
    if not fg:
        return False

    for existing in safe_list(primary_features):
        eg = safe_dict(existing.get("geo_bbox"))
        if not eg:
            continue
        overlap = overlap_ratio(fg, eg)
        dist = geo_distance(fg, eg)
        # Reject clear re-detections of the same primary water feature.
        if overlap > 0.62 or dist < 0.000035:
            return False

    # Gate out weak singleton ghost detections unless they carry former/disappearance language.
    if conf == "low" and len(years) <= 1:
        if not any(k in text for k in ["former", "infilled", "drained", "disappeared", "no longer visible", "historical"]):
            return False

    # Obvious adjacent/outside wording is not strong enough to carry through hunter mode.
    if conf == "low" and any(k in text for k in ["outside", "outside boundary", "east-northeast boundary", "adjacent"]):
        return False

    return True

def needs_secondary_scan(features: List[Dict[str, Any]]) -> bool:
    water_features = [
        f for f in features
        if safe_str(f.get("feature_type"), "other") in ("former_pond", "probable_pond")
    ]
    return len(water_features) <= 1


def should_run_followup(analysis: Dict[str, Any]) -> bool:
    findings = ensure_findings_dict(analysis.get("historical_findings"))
    ponds = safe_dict(findings.get("former_ponds_dams"))
    fill = safe_dict(findings.get("fill_or_disturbance"))

    return (
        ponds.get("status") in ("possible", "likely", "strong_evidence")
        or fill.get("status") in ("possible", "likely", "strong_evidence")
        or len(get_on_site_water_features(analysis)) >= 1
    )


def should_run_remainder_rescan(analysis: Dict[str, Any], polygon_present: bool) -> bool:
    if not polygon_present:
        return False
    return needs_secondary_scan(get_on_site_water_features(analysis))


def merge_unique_feature_lists(base_features: List[Dict[str, Any]], extra_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()

    for item in base_features + extra_features:
        key = (
            safe_str(item.get("feature_type"), "other").lower(),
            safe_str(item.get("location_relation"), "on_site").lower(),
            safe_str(item.get("notes"), "").strip().lower()[:90],
        )
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def merge_analyses(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    resolved: Optional[Dict[str, Any]] = None,
    image_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    merged = dict(primary)

    primary_features = [f for f in safe_list(primary.get("distinct_features")) if isinstance(f, dict)]
    secondary_features = [f for f in safe_list(secondary.get("distinct_features")) if isinstance(f, dict)]
    merged["distinct_features"] = merge_feature_sets_geometry_first(primary_features, secondary_features)
    merged["distinct_features"] = preserve_primary_historical_water_features(
        primary_features,
        merged["distinct_features"],
    )

    for key in [
        "visible_observations",
        "possible_risks",
        "recommended_follow_up",
        "recommended_investigation_focus",
        "limitations",
    ]:
        a = [str(x) for x in safe_list(primary.get(key)) if str(x).strip()]
        b = [str(x) for x in safe_list(secondary.get(key)) if str(x).strip()]
        merged[key] = list(dict.fromkeys(a + b))

    merged["feature_timeline"] = normalize_feature_timeline_items(
        safe_list(primary.get("feature_timeline")) + safe_list(secondary.get("feature_timeline"))
    )
    merged["change_timeline"] = normalize_change_timeline_items(
        safe_list(primary.get("change_timeline")) + safe_list(secondary.get("change_timeline"))
    )

    merged_findings = ensure_findings_dict(primary.get("historical_findings"))
    secondary_findings = ensure_findings_dict(secondary.get("historical_findings"))

    severity_rank = {
        "none": 0,
        "minor": 1,
        "possible": 2,
        "moderate": 2,
        "likely": 3,
        "major": 3,
        "strong_evidence": 4,
    }

    for key in ["former_ponds_dams", "vegetation_clearing", "fill_or_disturbance"]:
        a = safe_dict(merged_findings.get(key))
        b = safe_dict(secondary_findings.get(key))
        status_a = safe_str(a.get("status"), "none")
        status_b = safe_str(b.get("status"), "none")
        merged_findings[key] = a if severity_rank.get(status_a, 0) >= severity_rank.get(status_b, 0) else b

    merged["historical_findings"] = merged_findings
    # Do not sanitize here. Final truth-layer sanitize runs once at the end.
    return merged

def status_chip_text(status: str) -> str:
    mapping = {
        "none": "Low",
        "minor": "Low",
        "possible": "Moderate",
        "moderate": "Moderate",
        "likely": "Elevated",
        "major": "Elevated",
        "strong_evidence": "High",
    }
    return mapping.get((status or "").lower(), "Review")


def fetch_image_bytes(url: str) -> Optional[bytes]:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def fetch_image_bytes_with_opacity(url: str, opacity: float = 0.30) -> Optional[bytes]:
    raw = fetch_image_bytes(url)
    if not raw:
        return None

    try:
        from PIL import Image as PILImage
        img = PILImage.open(BytesIO(raw)).convert("RGBA")
        alpha = img.getchannel("A")
        alpha = alpha.point(lambda p: int(p * max(0.0, min(1.0, opacity))))
        img.putalpha(alpha)
        output = BytesIO()
        white_bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
        composited = PILImage.alpha_composite(white_bg, img)
        composited.convert("RGB").save(output, format="PNG")
        return output.getvalue()
    except Exception:
        return raw


def build_mapbox_streets_bbox_url(bbox: Dict[str, float], size: str = "900x620") -> str:
    if not MAPBOX_TOKEN:
        return ""
    return (
        f"{MAPBOX_BASE_URL}/streets-v12/static/"
        f"[{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}]"
        f"/{size}?padding=0&access_token={MAPBOX_TOKEN}"
    )


def fetch_geology_mapbox_composite_image(geology_image: Dict[str, Any]) -> Optional[bytes]:
    """Mapbox streets below + full-strength geology over it for the PDF geology figure."""
    geology_url = safe_str(geology_image.get("url"), "")
    bbox = safe_dict(geology_image.get("bbox"))
    geology_raw = fetch_image_bytes(geology_url)
    if not geology_raw:
        return None

    basemap_raw = fetch_image_bytes(build_mapbox_streets_bbox_url(bbox)) if bbox else None
    if not basemap_raw:
        return geology_raw

    try:
        from PIL import Image as PILImage
        base = PILImage.open(BytesIO(basemap_raw)).convert("RGBA")
        geology = PILImage.open(BytesIO(geology_raw)).convert("RGBA").resize(base.size)

        # Geology at native/full opacity over the street basemap.
        composed = PILImage.alpha_composite(base, geology)

        # Subtle street/label pass to keep road context visible, QLD-Globe style.
        road_overlay = base.copy()
        road_overlay.putalpha(62)
        composed = PILImage.alpha_composite(composed, road_overlay)

        output = BytesIO()
        composed.convert("RGB").save(output, format="PNG")
        return output.getvalue()
    except Exception:
        return geology_raw


def compact_report_address(address: str) -> str:
    """Shorten long Mapbox addresses so the cover image stays on page 1."""
    value = safe_str(address, "").strip()
    if not value:
        return value

    value = re.sub(r",?\s*Australia\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bQueensland\b", "QLD", value, flags=re.IGNORECASE)
    value = re.sub(r"\bNew South Wales\b", "NSW", value, flags=re.IGNORECASE)
    value = re.sub(r"\bVictoria\b", "VIC", value, flags=re.IGNORECASE)
    value = re.sub(r"\bSouth Australia\b", "SA", value, flags=re.IGNORECASE)
    value = re.sub(r"\bWestern Australia\b", "WA", value, flags=re.IGNORECASE)
    value = re.sub(r"\bTasmania\b", "TAS", value, flags=re.IGNORECASE)
    value = re.sub(r"\bNorthern Territory\b", "NT", value, flags=re.IGNORECASE)
    value = re.sub(r"\bAustralian Capital Territory\b", "ACT", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"\s+,", ",", value)
    return value.strip(" ,")


def strip_mapbox_polygon_overlay(url: str) -> str:
    url = safe_str(url, "")
    if "/static/geojson(" not in url:
        return url
    try:
        prefix, rest = url.split("/static/geojson(", 1)
        _, suffix = rest.split(")/", 1)
        return prefix + "/static/" + suffix
    except Exception:
        return url


def report_display_url(image: Optional[Dict[str, Any]], keep_boundary_overlay: bool = False) -> str:
    if not image:
        return ""
    url = safe_str(image.get("url"), "")
    if keep_boundary_overlay:
        return url
    if safe_str(image.get("source"), "").startswith("mapbox_"):
        return strip_mapbox_polygon_overlay(url)
    return url



def draw_brand_lockup(canvas, x: float, y: float, text_value: str, fill_color: str, width_mm: float = 28):
    width_pt = width_mm * mm
    height_pt = 8.5 * mm
    canvas.setFillColor(colors.HexColor(fill_color))
    canvas.roundRect(x, y, width_pt, height_pt, 2.5 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont(REPORT_FONT_BOLD, 8.2)
    canvas.drawCentredString(x + (width_pt / 2), y + 2.6 * mm, text_value)


def draw_alpha_badge(canvas, x: float, y: float, text_value: str = ""):
    return



def draw_report_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    header_h = 20 * mm
    canvas.setFillColor(colors.HexColor("#162338"))
    canvas.rect(0, height - header_h, width, header_h, stroke=0, fill=1)

    badge_w = 1 * mm
    badge_h = 1 * mm
    badge_x = (width - badge_w) / 2
    badge_y = height - 13.6 * mm
    draw_alpha_badge(canvas, badge_x, badge_y, "")

    left_x = 10 * mm
    right_margin = 10 * mm
    logo_y = height - 16.5 * mm

    if DWGEO_LOGO_PATH and os.path.exists(DWGEO_LOGO_PATH):
        try:
            canvas.drawImage(
                DWGEO_LOGO_PATH,
                left_x,
                logo_y,
                width=44 * mm,
                height=11.5 * mm,
                preserveAspectRatio=True,
                mask='auto'
            )
        except Exception:
            draw_brand_lockup(canvas, left_x, height - 13.6 * mm, "DWGEO", "#0f172a", 30)
    else:
        draw_brand_lockup(canvas, left_x, height - 13.6 * mm, "DWGEO", "#0f172a", 30)

    right_logo_w = 44 * mm
    if SITECLASS_LOGO_PATH and os.path.exists(SITECLASS_LOGO_PATH):
        try:
            canvas.drawImage(
                SITECLASS_LOGO_PATH,
                width - right_margin - right_logo_w,
                logo_y,
                width=right_logo_w,
                height=11.5 * mm,
                preserveAspectRatio=True,
                mask='auto'
            )
        except Exception:
            draw_brand_lockup(canvas, width - right_margin - 42 * mm, height - 13.6 * mm, "SITE CLASS ONLINE", "#1e293b", 42)
    else:
        draw_brand_lockup(canvas, width - right_margin - 42 * mm, height - 13.6 * mm, "SITE CLASS ONLINE", "#1e293b", 42)

    canvas.setStrokeColor(colors.HexColor("#111111"))
    canvas.setLineWidth(1.0)
    canvas.line(18 * mm, 17 * mm, width - 18 * mm, 17 * mm)

    canvas.setFillColor(colors.HexColor("#6f7b90"))
    canvas.setFont(REPORT_FONT_REGULAR, 9)
    canvas.drawString(18 * mm, 10 * mm, "Generated by DWGEO")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")

    canvas.restoreState()


def build_pdf_styles():

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Heading1"],
        fontName=REPORT_FONT_BOLD,
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#162338"),
        spaceAfter=0,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="AlphaBadge",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubTitle",
        parent=styles["Normal"],
        fontName=REPORT_FONT_REGULAR,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#5b6577"),
        spaceAfter=6,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName=REPORT_FONT_BOLD,
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#162338"),
        spaceBefore=2,
        spaceAfter=6,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="SmallLabel",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#7a8596"),
        spaceAfter=2,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="BodyTextSmall",
        parent=styles["Normal"],
        fontName=REPORT_FONT_REGULAR,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#263244"),
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="BodyTextTight",
        parent=styles["Normal"],
        fontName=REPORT_FONT_REGULAR,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#263244"),
        spaceAfter=4,
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name="ChipText",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="TinyMuted",
        parent=styles["Normal"],
        fontName=REPORT_FONT_REGULAR,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#788395"),
        spaceAfter=2,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="DisclaimerTitle",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#162338"),
        alignment=TA_CENTER,
        spaceAfter=3,
    ))

    styles.add(ParagraphStyle(
        name="DisclaimerBody",
        parent=styles["Normal"],
        fontName=REPORT_FONT_REGULAR,
        fontSize=8.8,
        leading=12,
        textColor=colors.HexColor("#334155"),
        alignment=TA_LEFT,
        spaceAfter=0,
    ))

    styles.add(ParagraphStyle(
        name="BoxHeading",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#162338"),
        alignment=TA_LEFT,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="RiskCardTitle",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#162338"),
        alignment=TA_LEFT,
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["Normal"],
        fontName=REPORT_FONT_BOLD,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#162338"),
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["Normal"],
        fontName=REPORT_FONT_REGULAR,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#263244"),
        spaceAfter=0,
        alignment=TA_LEFT,
    ))

    return styles


def make_status_table(title: str, status: str, notes: str, styles) -> Table:
    status_text = status_chip_text(status)
    status_lower = (status or "").lower()

    if status_lower in ("strong_evidence",):
        chip_color = colors.HexColor("#d9534f")
    elif status_lower in ("likely", "major"):
        chip_color = colors.HexColor("#e38b2c")
    elif status_lower in ("possible", "moderate"):
        chip_color = colors.HexColor("#d7ac52")
    else:
        chip_color = colors.HexColor("#4caf7d")

    chip = Table(
        [[Paragraph(status_text, styles["ChipText"])]],
        colWidths=[24 * mm],
        rowHeights=[9 * mm]
    )
    chip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), chip_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0, chip_color),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    title_para = Paragraph(f"<b>{title}</b>", styles["BodyTextSmall"])
    notes_para = Paragraph(notes or "No significant indicators noted from current automated screening.", styles["BodyTextSmall"])

    inner = Table(
        [[title_para, chip], [notes_para, ""]],
        colWidths=[130 * mm, 32 * mm]
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#d8dee8")),
        ("INNERGRID", (0, 0), (-1, -1), 0, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("SPAN", (0, 1), (1, 1)),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
        ("RIGHTPADDING", (0, 0), (0, -1), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 0),
        ("RIGHTPADDING", (1, 0), (1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return centered_flowable(inner, total_width_mm=170)



def centered_flowable(flowable: Flowable, total_width_mm: float = 170) -> Table:
    tbl = Table([[flowable]], colWidths=[total_width_mm * mm])
    tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return tbl


class UnderlinedHeadingFlowable(Flowable):
    def __init__(
        self,
        text: str,
        font_name: str,
        font_size: float,
        text_color,
        line_color,
        line_width: float = 0.8,
        extend_mm: float = 6.0,
        gap_pt: float = 5.0,
        beta_text: str = "",
        beta_font_name: Optional[str] = None,
        beta_font_size: float = 10.0,
        beta_color=None,
        beta_gap_pt: float = 4.0,
        extra_bottom_pt: float = 8.0,
    ):
        super().__init__()
        self.text = text
        self.font_name = font_name
        self.font_size = font_size
        self.text_color = text_color
        self.line_color = line_color
        self.line_width = line_width
        self.extend_pt = extend_mm * mm
        self.gap_pt = gap_pt
        self.beta_text = beta_text
        self.beta_font_name = beta_font_name or font_name
        self.beta_font_size = beta_font_size
        self.beta_color = beta_color or text_color
        self.beta_gap_pt = beta_gap_pt
        self.extra_bottom_pt = extra_bottom_pt
        self._avail_width = 0
        self._text_width = 0
        self._beta_width = 0
        self._group_width = 0
        self._text_baseline_y = 0

    def wrap(self, availWidth, availHeight):
        self._avail_width = availWidth
        self._text_width = stringWidth(self.text, self.font_name, self.font_size)
        self._beta_width = stringWidth(self.beta_text, self.beta_font_name, self.beta_font_size) if self.beta_text else 0
        self._group_width = self._text_width + (self.beta_gap_pt if self.beta_text else 0) + self._beta_width
        self._text_baseline_y = self.line_width + self.gap_pt + 1.0
        height = max(self.font_size, self.beta_font_size) + self._text_baseline_y + self.extra_bottom_pt
        return availWidth, height

    def draw(self):
        c = self.canv
        start_x = max(0.0, (self._avail_width - self._group_width) / 2.0)
        y = self._text_baseline_y

        c.setFillColor(self.text_color)
        c.setFont(self.font_name, self.font_size)
        c.drawString(start_x, y, self.text)

        if self.beta_text:
            beta_x = start_x + self._text_width + self.beta_gap_pt
            beta_y = y + max(0.0, (self.font_size - self.beta_font_size) * 0.18)
            c.setFillColor(self.beta_color)
            c.setFont(self.beta_font_name, self.beta_font_size)
            c.drawString(beta_x, beta_y, self.beta_text)

        c.setStrokeColor(self.line_color)
        c.setLineWidth(self.line_width)
        line_y = 0.8
        c.line(
            max(0.0, start_x - self.extend_pt),
            line_y,
            min(self._avail_width, start_x + self._group_width + self.extend_pt),
            line_y,
        )


def make_underlined_heading(text_value: str, styles, title: bool = False, beta: bool = False) -> Flowable:
    if title:
        return UnderlinedHeadingFlowable(
            text=text_value,
            font_name=REPORT_FONT_BOLD,
            font_size=24,
            text_color=colors.HexColor("#162338"),
            line_color=colors.HexColor("#d7ac52"),
            line_width=1.0,
            extend_mm=7.0,
            gap_pt=4.0,
            beta_text="(Beta)" if beta else "",
            beta_font_name=REPORT_FONT_SEMIBOLD,
            beta_font_size=11.5,
            beta_color=colors.HexColor("#6b7280"),
            beta_gap_pt=5.0,
            extra_bottom_pt=12.0,
        )
    return Paragraph(text_value, styles["SectionHeading"])


def make_disclaimer_box(styles) -> Table:
    title = Paragraph("Preliminary Screening Notice", styles["DisclaimerTitle"])
    body = Paragraph(
        "This report is generated using automated interpretation of available aerial imagery and is intended for preliminary site screening purposes only.<br/><br/>"
        "While reasonable effort has been made to identify visible indicators such as ponds, historical water features, and site disturbance, not all features may be detected or accurately interpreted.<br/><br/>"
        "Users are strongly encouraged to independently review historical imagery (e.g. Google Earth) and obtain a detailed geotechnical investigation in accordance with AS2870 prior to design, construction, or purchase decisions.",
        styles["DisclaimerBody"]
    )
    inner = Table([[title], [body]], colWidths=[162 * mm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d9dee6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    outer = Table([[inner]], colWidths=[170 * mm])
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def make_alpha_badge(styles) -> Table:
    pill = Table([[""]], colWidths=[1 * mm], rowHeights=[1 * mm])
    pill.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0, colors.white),
    ]))
    return pill

def make_site_details_table(payload: SiteRequest, resolved: Dict[str, Any], styles) -> Table:
    rows = [
        [Paragraph("<b>Address</b>", styles["BodyTextTight"]), Paragraph(safe_str(resolved.get("matched_address"), payload.address) or payload.address, styles["BodyTextTight"])],
        [Paragraph("<b>Location source</b>", styles["BodyTextTight"]), Paragraph(str(resolved.get("location_source", "")), styles["BodyTextTight"])],
        [Paragraph("<b>Latitude / Longitude</b>", styles["BodyTextTight"]), Paragraph(f"{resolved.get('lat'):.6f}, {resolved.get('lng'):.6f}", styles["BodyTextTight"])],
        [Paragraph("<b>Site bbox</b>", styles["BodyTextTight"]), Paragraph(f"{resolved.get('bbox_width_m')} m × {resolved.get('bbox_height_m')} m", styles["BodyTextTight"])],
        [Paragraph("<b>Context bbox</b>", styles["BodyTextTight"]), Paragraph(f"{resolved.get('context_bbox_width_m')} m × {resolved.get('context_bbox_height_m')} m", styles["BodyTextTight"])],
        [Paragraph("<b>Polygon supplied</b>", styles["BodyTextTight"]), Paragraph("Yes" if resolved.get("polygon") else "No", styles["BodyTextTight"])],
    ]

    tbl = Table(rows, colWidths=[44 * mm, 116 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fbfd")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#d8dee8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e8ef")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def make_key_flags_table(findings: Dict[str, Any], styles) -> Table:
    former_ponds = safe_dict(findings.get("former_ponds_dams"))
    vegetation = safe_dict(findings.get("vegetation_clearing"))
    fill_disturbance = safe_dict(findings.get("fill_or_disturbance"))

    rows = [
        [
            Paragraph("<b>Former water features</b>", styles["BodyTextTight"]),
            Paragraph(status_chip_text(former_ponds.get("status", "possible")), styles["BodyTextTight"]),
            Paragraph(safe_str(former_ponds.get("notes"), "No notes."), styles["BodyTextTight"]),
        ],
        [
            Paragraph("<b>Vegetation / land change</b>", styles["BodyTextTight"]),
            Paragraph(status_chip_text(vegetation.get("status", "minor")), styles["BodyTextTight"]),
            Paragraph(safe_str(vegetation.get("notes"), "No notes."), styles["BodyTextTight"]),
        ],
        [
            Paragraph("<b>Fill / disturbance</b>", styles["BodyTextTight"]),
            Paragraph(status_chip_text(fill_disturbance.get("status", "possible")), styles["BodyTextTight"]),
            Paragraph(safe_str(fill_disturbance.get("notes"), "No notes."), styles["BodyTextTight"]),
        ],
    ]

    tbl = Table(
        [[Paragraph("<b>Key Flag</b>", styles["BodyTextTight"]),
          Paragraph("<b>Level</b>", styles["BodyTextTight"]),
          Paragraph("<b>Comment</b>", styles["BodyTextTight"])]
         ] + rows,
        colWidths=[48 * mm, 24 * mm, 88 * mm]
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f7")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f9fbfd")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#d8dee8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e8ef")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def feature_buckets(features: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    primary = []
    secondary = []
    contextual = []

    for f in features:
        bucket = safe_str(f.get("risk_priority"), "secondary")
        if bucket == "primary":
            primary.append(f)
        elif bucket == "contextual":
            contextual.append(f)
        else:
            secondary.append(f)

    return {"primary": primary, "secondary": secondary, "contextual": contextual}


def make_distinct_features_table(features: List[Dict[str, Any]], styles) -> Table:
    rows = [[
        Paragraph("<b>Feature</b>", styles["BodyTextTight"]),
        Paragraph("<b>Type</b>", styles["BodyTextTight"]),
        Paragraph("<b>Relation</b>", styles["BodyTextTight"]),
        Paragraph("<b>Confidence</b>", styles["BodyTextTight"]),
        Paragraph("<b>Notes</b>", styles["BodyTextTight"]),
    ]]

    for feature in features[:12]:
        rows.append([
            Paragraph(safe_str(feature.get("feature_id"), "Unnamed"), styles["BodyTextTight"]),
            Paragraph(safe_str(feature.get("feature_type"), "other"), styles["BodyTextTight"]),
            Paragraph(safe_str(feature.get("location_relation"), "unclear"), styles["BodyTextTight"]),
            Paragraph(safe_str(feature.get("confidence"), "low"), styles["BodyTextTight"]),
            Paragraph(safe_str(feature.get("notes"), ""), styles["BodyTextTight"]),
        ])

    tbl = Table(rows, colWidths=[28 * mm, 26 * mm, 26 * mm, 24 * mm, 52 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f7")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f9fbfd")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#d8dee8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e8ef")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl



def select_brief_report_features(features: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    on_site_former = [
        f for f in features
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") == "former_pond"
    ]
    on_site_current = [
        f for f in features
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") == "pond"
    ]
    disturbances = [
        f for f in features
        if safe_str(f.get("location_relation"), "") == "on_site"
        and safe_str(f.get("feature_type"), "") in ("disturbance", "fill_area")
    ]

    def sort_feats(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda f: (
                confidence_rank(safe_str(f.get("confidence"), "low")),
                len([y for y in safe_list(f.get("detected_in_years")) if isinstance(y, (int, float))]),
                len(safe_str(f.get("notes"), "")),
            ),
            reverse=True,
        )

    return {
        "historical_water": sort_feats(on_site_former)[:2] or sort_feats(on_site_current)[:1],
        "current_water": sort_feats(on_site_current)[:1],
        "disturbance": sort_feats(disturbances)[:1],
    }

def choose_report_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    def first_by_label(label: str):
        for img in images:
            if img.get("label") == label:
                return img
        return None

    hist = sorted(
        [img for img in images if img.get("type") == "historical"],
        key=lambda x: (x.get("year") if isinstance(x.get("year"), int) else 9999, x.get("label", "")),
    )

    current_img = first_by_label("current_site") or first_by_label("current_context") or first_by_label("current_wide_context")
    earliest_hist = hist[0] if hist else None
    latest_hist = hist[-1] if hist else None

    for candidate in [current_img, earliest_hist, latest_hist]:
        if candidate and all(candidate.get("label") != existing.get("label") for existing in selected):
            selected.append(candidate)

    return selected[:3]

def choose_boundary_overview_image(images: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for label in ["current_site", "current_context", "current_wide_context"]:
        for img in images:
            if img.get("label") == label:
                return img
    return images[0] if images else None


def annotations_for_image(features: List[Dict[str, Any]], image_label: str, image_bbox: Dict[str, float], hero_mode: bool = False) -> List[Dict[str, Any]]:
    out = []

    for f in features:
        detected_label = safe_str(f.get("detected_on_image"), "")
        primary_label = safe_str(f.get("primary_image_label"), "")
        bbox = safe_list(f.get("approximate_bbox_norm"))

        if hero_mode:
            continue

        use_bbox: List[float] = []

        if image_label == detected_label or image_label == primary_label:
            if len(bbox) == 4:
                use_bbox = clamp_norm_bbox(bbox)

        if not use_bbox:
            geo_bbox = safe_dict(f.get("geo_bbox"))
            reproj = geo_bbox_to_norm_bbox(geo_bbox, image_bbox) if geo_bbox else None
            if reproj:
                if image_label.startswith("historical_"):
                    use_bbox = clamp_norm_bbox(reproj)
                elif image_label.startswith("current_") and safe_str(f.get("location_relation"), "") == "on_site":
                    use_bbox = clamp_norm_bbox(reproj)

        if len(use_bbox) != 4:
            continue

        if safe_str(f.get("location_relation"), "") != "on_site" and image_label.startswith("current_"):
            continue

        copy_f = dict(f)
        copy_f["approximate_bbox_norm"] = use_bbox
        out.append(copy_f)

    return out



class AnnotatedImageFlowable(Flowable):
    def __init__(self, img_bytes: bytes, annotations: List[Dict[str, Any]], width_mm: float = 160):
        super().__init__()
        self.img_bytes = img_bytes
        self.annotations = annotations or []
        self.width = width_mm * mm

        reader = ImageReader(BytesIO(img_bytes))
        iw, ih = reader.getSize()
        self.aspect = ih / iw if iw else 1.0
        self.height = self.width * self.aspect

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        c = self.canv
        reader = ImageReader(BytesIO(self.img_bytes))
        c.drawImage(reader, 0, 0, width=self.width, height=self.height, preserveAspectRatio=True, mask='auto')

        for ann in self.annotations:
            bbox = safe_list(ann.get("approximate_bbox_norm"))
            if len(bbox) != 4:
                continue

            try:
                x, y, w, h = [float(v) for v in bbox]
            except Exception:
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue

            rx = x * self.width
            ry = (1 - y - h) * self.height
            rw = w * self.width
            rh = h * self.height

            c.setStrokeColor(colors.HexColor("#d9534f"))
            c.setLineWidth(1.4)
            c.rect(rx, ry, rw, rh, stroke=1, fill=0)

            label = safe_str(ann.get("feature_id"), "").strip() or safe_str(ann.get("feature_type"), "Feature").replace("_", " ").title()
            font_name = REPORT_FONT_BOLD
            font_size = 8
            padding = 2

            label_w = stringWidth(label, font_name, font_size) + 2 * padding
            label_h = 10
            label_y = min(self.height - label_h, ry + rh)

            c.setFillColor(colors.HexColor("#d9534f"))
            c.rect(rx, label_y, label_w, label_h, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.setFont(font_name, font_size)
            c.drawString(rx + padding, label_y + 2, label)

        c.setFillColor(colors.black)



def make_title_details_table(payload: SiteRequest, resolved: Dict[str, Any], confidence_overall: str, styles) -> Table:
    matched_address = compact_report_address(resolved.get("matched_address") or payload.address)
    lot_area_m2 = polygon_area_m2(resolved.get("polygon"))
    lot_area_text = f"{lot_area_m2:,.0f} m²" if isinstance(lot_area_m2, (int, float)) else "Not available"

    rows = [
        [Paragraph("<b>Address</b>", styles["TableHeader"]), Paragraph(matched_address, styles["TableCell"])],
        [Paragraph("<b>Generated</b>", styles["TableHeader"]), Paragraph(datetime.now().strftime("%d %b %Y"), styles["TableCell"])],
        [Paragraph("<b>Lot size</b>", styles["TableHeader"]), Paragraph(lot_area_text, styles["TableCell"])],
        [Paragraph("<b>Confidence</b>", styles["TableHeader"]), Paragraph(confidence_overall.title(), styles["TableCell"])],
    ]
    tbl = Table(rows, colWidths=[32 * mm, 118 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d8dee8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e8ef")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return centered_flowable(tbl, total_width_mm=170)


def make_boxed_paragraph(title: str, body: str, styles, width_mm: float = 170) -> Table:
    inner = Table(
        [[Paragraph(title, styles["BoxHeading"])], [Paragraph(body, styles["BodyTextSmall"])]],
        colWidths=[(width_mm - 8) * mm]
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d8dee8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return centered_flowable(inner, total_width_mm=width_mm)


def make_image_figure(image_bytes: bytes, annotations: List[Dict[str, Any]], caption: Optional[str], width_mm: float = 160) -> Table:
    flow = AnnotatedImageFlowable(image_bytes, annotations, width_mm=width_mm)
    rows = [[flow]]
    if caption:
        rows.append([Paragraph(caption, build_pdf_styles()["TinyMuted"])])
    tbl = Table(rows, colWidths=[170 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, 0), 0.8, colors.HexColor("#d8dee8")),
        ("LEFTPADDING", (0, 0), (-1, 0), 5),
        ("RIGHTPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return tbl



def make_simple_box(body: str, styles, width_mm: float = 170) -> Table:
    body_para = Paragraph(body or "", styles["BodyTextSmall"])
    inner = Table(
        [[body_para]],
        colWidths=[(width_mm - 4) * mm],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d9dee6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    outer = Table([[inner]], colWidths=[width_mm * mm])
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def make_figure_panel(image_flowable: Flowable, caption: str, styles, width_mm: float = 170) -> Table:
    caption_para = Paragraph(caption, styles["TinyMuted"])
    inner = Table([[image_flowable], [caption_para]], colWidths=[(width_mm - 4) * mm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d9dee6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    outer = Table([[inner]], colWidths=[width_mm * mm])
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def make_risk_card(risk: Dict[str, str], styles, width_mm: float = 170) -> Table:
    level = safe_str(risk.get("level"), "Review")
    level_color = "#d9534f" if level.upper() == "HIGH" else "#e38b2c" if level.upper() == "ELEVATED" else "#d7ac52"
    chip = Table([[Paragraph(level, styles["ChipText"])]], colWidths=[24 * mm], rowHeights=[8 * mm])
    chip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(level_color)),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("BOX", (0, 0), (-1, -1), 0, colors.HexColor(level_color)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    title = Paragraph(safe_str(risk.get("title"), "Risk"), styles["RiskCardTitle"])
    body = Paragraph(safe_str(risk.get("text"), ""), styles["BodyTextSmall"])
    inner = Table([[title, chip], [body, ""]], colWidths=[130 * mm, 32 * mm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d9dee6")),
        ("SPAN", (0, 1), (1, 1)),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (0, -1), 10),
        ("RIGHTPADDING", (0, 0), (0, -1), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 0),
        ("RIGHTPADDING", (1, 0), (1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    outer = Table([[inner]], colWidths=[width_mm * mm])
    outer.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer

def build_report_pdf(
    payload: SiteRequest,
    resolved: Dict[str, Any],
    images: List[Dict[str, Any]],
    analysis: Dict[str, Any]
) -> str:
    styles = build_pdf_styles()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = sanitize_filename(payload.address)
    filename = f"{base_name}-{timestamp}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = BaseDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=24 * mm,
        bottomMargin=18 * mm,
        title="AI Site History Report",
        author="DWGEO",
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    template = PageTemplate(id="report", frames=[frame], onPage=draw_report_header_footer)
    doc.addPageTemplates([template])

    findings = ensure_findings_dict(analysis.get("historical_findings"))
    summary = safe_str(analysis.get("summary"), "No summary generated.")
    outcome = safe_str(analysis.get("screening_outcome"), "Detailed geotechnical investigation is strongly recommended.")
    limitations = safe_list(analysis.get("limitations"))
    confidence_overall = safe_str(analysis.get("confidence_overall"), "low")
    distinct_features = safe_list(analysis.get("distinct_features"))
    geotechnical_risks = [r for r in safe_list(analysis.get("geotechnical_risks")) if isinstance(r, dict)]
    surface_geology_context = build_surface_geology_context(resolved)
    surface_geology_text = format_surface_geology_context(surface_geology_context)
    surface_geology_text = surface_geology_text or (
        "Mapped surface regional geology could not be automatically retrieved for this site from the available public geology service.<br/><br/>"
        "Mapped geology is provided for regional context only and does not replace intrusive geotechnical investigation.<br/><br/>"
        "A detailed geotechnical investigation is required to confirm actual ground conditions and site classification in accordance with AS2870."
    )
    surface_geology_image = build_surface_geology_context_image(resolved)

    matched_address = compact_report_address(resolved.get("matched_address") or payload.address)
    lot_area_m2 = polygon_area_m2(resolved.get("polygon"))
    lot_area_text = f"{lot_area_m2:,.0f} m²" if isinstance(lot_area_m2, (int, float)) else "Not available"
    years_scanned = sorted({int(img.get("year")) for img in images if isinstance(img.get("year"), int)})
    years_scanned_text = ", ".join(str(y) for y in years_scanned) if years_scanned else "Current imagery only"

    brief = select_brief_report_features(distinct_features)
    selected_images = choose_report_images(images)
    boundary_overview_img = choose_boundary_overview_image(images)

    current_img = selected_images[0] if len(selected_images) > 0 else None
    historical_img = selected_images[1] if len(selected_images) > 1 else None
    later_img = selected_images[2] if len(selected_images) > 2 else None

    story = []

    # Page 1
    story.append(Spacer(1, 4 * mm))
    story.append(make_underlined_heading("AI SITE HISTORY REPORT", styles, title=True, beta=True))
    story.append(Spacer(1, 2.2 * mm))
    details_rows = [
        [Paragraph("<b>Address</b>", styles["TableHeader"]), Paragraph(matched_address, styles["TableCell"])],
        [Paragraph("<b>Generated</b>", styles["TableHeader"]), Paragraph(datetime.now().strftime('%d %b %Y'), styles["BodyTextSmall"])],
        [Paragraph("<b>Lot size</b>", styles["TableHeader"]), Paragraph(lot_area_text, styles["BodyTextSmall"])],
        [Paragraph("<b>Years scanned</b>", styles["TableHeader"]), Paragraph(years_scanned_text, styles["BodyTextSmall"])],
        [Paragraph("<b>Confidence</b>", styles["TableHeader"]), Paragraph(confidence_overall.title(), styles["BodyTextSmall"])],
    ]
    details_tbl = Table(details_rows, colWidths=[30 * mm, 128 * mm])
    details_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d9dee6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e8ef")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(centered_flowable(details_tbl, total_width_mm=170))
    story.append(Spacer(1, 3 * mm))
    story.append(make_disclaimer_box(styles))
    story.append(Spacer(1, 2.5 * mm))

    if boundary_overview_img:
        story.append(make_underlined_heading("Current Site Overview", styles))
        image_bytes = fetch_image_bytes(report_display_url(boundary_overview_img, keep_boundary_overlay=True))
        if image_bytes:
            fig = centered_flowable(AnnotatedImageFlowable(image_bytes, [], width_mm=88), total_width_mm=160)
            story.append(make_figure_panel(fig, "Figure 1: Current site overview used for automated screening.", styles, width_mm=170))
        story.append(Spacer(1, 2 * mm))

    story.append(make_underlined_heading("Executive Summary", styles))
    story.append(make_simple_box(summary, styles, width_mm=170))
    story.append(Spacer(1, 4 * mm))

    story.append(make_underlined_heading("Underlying Surface Regional Geology", styles))
    if surface_geology_image and surface_geology_image.get("url"):
        geology_img_bytes = fetch_geology_mapbox_composite_image(surface_geology_image)
        if geology_img_bytes:
            try:
                story.append(centered_flowable(
                    Image(BytesIO(geology_img_bytes), width=150 * mm, height=78 * mm),
                    total_width_mm=150
                ))
                story.append(Paragraph(
                    "Figure: Regional mapped surface geology context around the site. Geological mapping is provided for context only.",
                    styles["TinyMuted"]
                ))
                story.append(Spacer(1, 3 * mm))
            except Exception:
                pass
    story.append(make_simple_box(surface_geology_text, styles, width_mm=170))
    story.append(Spacer(1, 2.5 * mm))

    # Page 2
    story.append(make_underlined_heading("Key Site Risks", styles))

    has_current_pond = any(
        safe_str(f.get("feature_type"), "") == "pond"
        and safe_str(f.get("location_relation"), "") == "on_site"
        for f in distinct_features
    )
    has_pond_risk = any(
        safe_str(r.get("title"), "") == "Pond / Wet Area Risk"
        for r in geotechnical_risks
    )

    if has_current_pond or has_pond_risk:
        story.append(make_status_table(
            "Current Pond / Wet Area",
            "strong_evidence",
            "A current pond or pond-like depression is present on-site and should be treated as a relevant moisture-related geotechnical risk.",
            styles
        ))
        story.append(Spacer(1, 3 * mm))

    former_findings = safe_dict(findings.get("former_ponds_dams"))
    if safe_str(former_findings.get("status"), "none") != "none":
        story.append(make_status_table(
            "Former Water Features",
            former_findings.get("status", "possible"),
            former_findings.get("notes", ""),
            styles
        ))
        story.append(Spacer(1, 3 * mm))

    story.append(make_status_table(
        "Fill / Disturbance",
        safe_dict(findings.get("fill_or_disturbance")).get("status", "possible"),
        safe_dict(findings.get("fill_or_disturbance")).get("notes", ""),
        styles
    ))
    story.append(Spacer(1, 2.5 * mm))
    story.append(KeepTogether([
        make_underlined_heading("Recommendation", styles),
        make_simple_box(outcome, styles, width_mm=170),
    ]))

    # Page 3
    if current_img:
        story.append(PageBreak())
        story.append(make_underlined_heading("Current Site Image", styles))
        story.append(Paragraph(
            "North-up current imagery with the submitted lot boundary shown for site reference.",
            styles["BodyTextSmall"]
        ))
        image_bytes = fetch_image_bytes(report_display_url(current_img, keep_boundary_overlay=True))
        if image_bytes:
            anns = []
            fig = centered_flowable(AnnotatedImageFlowable(image_bytes, anns, width_mm=160), total_width_mm=160)
            story.append(make_figure_panel(fig, f"Source: {current_img.get('source', 'unknown')}", styles, width_mm=170))

    # Page 4
    if geotechnical_risks:
        story.append(PageBreak())
        story.append(make_underlined_heading("Geotechnical Risk Summary", styles))
        story.append(Paragraph(
            "The following preliminary risk commentary is based on interpreted site history features identified from available aerial imagery.",
            styles["BodyTextSmall"]
        ))
        story.append(Spacer(1, 2 * mm))
        for idx, risk in enumerate(geotechnical_risks):
            story.append(make_risk_card(risk, styles, width_mm=170))
            if idx < len(geotechnical_risks) - 1:
                story.append(Spacer(1, 3 * mm))

    # Page 5 historical evidence
    if historical_img:
        story.append(PageBreak())
        story.append(make_underlined_heading("Historical Evidence", styles))
        story.append(Paragraph(
            "Representative historical imagery selected for site-history interpretation. Annotation boxes have been removed for cleaner presentation.",
            styles["BodyTextSmall"]
        ))
        image_bytes = fetch_image_bytes(historical_img.get("url", ""))
        if image_bytes:
            anns = []
            fig = centered_flowable(AnnotatedImageFlowable(image_bytes, anns, width_mm=160), total_width_mm=160)
            caption_bits = [f"Source: {historical_img.get('source', 'unknown')}"]
            if historical_img.get("scene_title"):
                caption_bits.append(str(historical_img.get("scene_title")))
            if historical_img.get("capture_date"):
                caption_bits.append(str(historical_img.get("capture_date")))
            story.append(make_figure_panel(fig, " | ".join(caption_bits), styles, width_mm=170))

    # Page 6 later comparison - intentionally no annotations
    if later_img:
        story.append(PageBreak())
        story.append(make_underlined_heading("Later Comparison Image", styles))
        story.append(Paragraph(
            "Later imagery used to compare site conditions and assess whether historical features remain visible or appear altered.",
            styles["BodyTextSmall"]
        ))
        image_bytes = fetch_image_bytes(later_img.get("url", ""))
        if image_bytes:
            fig = centered_flowable(AnnotatedImageFlowable(image_bytes, [], width_mm=160), total_width_mm=160)
            caption_bits = [f"Source: {later_img.get('source', 'unknown')}"]
            if later_img.get("scene_title"):
                caption_bits.append(str(later_img.get("scene_title")))
            if later_img.get("capture_date"):
                caption_bits.append(str(later_img.get("capture_date")))
            story.append(make_figure_panel(fig, " | ".join(caption_bits), styles, width_mm=170))

    # Final page
    story.append(PageBreak())
    story.append(make_underlined_heading("Limitations & Disclaimer", styles))
    limit_text = "<br/>".join([f"• {str(item)}" for item in limitations[:6]]) if limitations else "• This report is based on visual review of available aerial imagery only."
    story.append(make_underlined_heading("Limitations", styles))
    story.append(make_simple_box(limit_text, styles, width_mm=170))
    story.append(Spacer(1, 4 * mm))
    story.append(make_underlined_heading("Disclaimer", styles))
    story.append(make_simple_box(
        "This AI site history scan is a preliminary screening tool only. It is intended to identify possible visual indicators for further professional review. It must not be relied on as a substitute for geotechnical investigation, engineering assessment, or design advice.",
        styles,
        width_mm=170,
    ))

    doc.build(story)
    return filename

@app.post("/analyze-site-ai")
def analyze_site_ai(payload: SiteRequest, request: Request):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found.")
    if not MAPBOX_TOKEN:
        raise HTTPException(status_code=500, detail="MAPBOX_TOKEN not found.")

    resolved = resolve_site_geometry(payload)
    lat = resolved["lat"]
    lng = resolved["lng"]
    bbox = resolved["bbox"]
    context_bbox = resolved["context_bbox"]
    wide_context_bbox = resolved["wide_context_bbox"]
    subzones = resolved["subzones"]
    polygon = resolved.get("polygon")

    current_images = build_current_mapbox_images(
        lat=lat,
        lng=lng,
        bbox=bbox,
        context_bbox=context_bbox,
        wide_context_bbox=wide_context_bbox,
        subzones=subzones,
        polygon=polygon,
    )

    historical_bundle = build_historical_qld_images(
        lat=lat,
        lng=lng,
        bbox=bbox,
        context_bbox=context_bbox,
        subzones=subzones,
        max_scenes=MAX_INITIAL_SCENES,
    )

    initial_images = current_images + historical_bundle["images"]
    water_signal_initial = simple_water_indicator(initial_images)

    initial_analysis = call_ai_screening(
        payload=payload,
        resolved=resolved,
        images=initial_images,
        water_signal=water_signal_initial,
        mode="initial",
        primary_features=None,
    )

    followup_ran = False
    followup_images: List[Dict[str, Any]] = []
    followup_analysis: Optional[Dict[str, Any]] = None
    followup_years: List[int] = []

    if should_run_followup(initial_analysis):
        followup_scenes = pick_followup_scenes(
            all_candidates=historical_bundle["all_candidates"],
            initial_selected=historical_bundle["selected_scenes"],
            analysis=initial_analysis,
            max_scenes=MAX_FOLLOWUP_SCENES,
        )

        if followup_scenes:
            followup_years = [s["year"] for s in followup_scenes]

            obvious_current_pond_exists, buried_secondary_risk_exists = classify_scene_budget_priority(initial_analysis)

            followup_hist_images = build_historical_images_from_scenes(
                scenes=followup_scenes,
                bbox=bbox,
                context_bbox=context_bbox,
                subzones=subzones,
                label_prefix="historical_qld_followup",
                include_context_for_edge_years=True,
                include_subzones_for_edge_years=True,
                prioritize_subzones=buried_secondary_risk_exists,
            )

            followup_images = current_images + followup_hist_images
            water_signal_followup = simple_water_indicator(followup_images)

            followup_analysis = call_ai_screening(
                payload=payload,
                resolved=resolved,
                images=followup_images,
                water_signal=water_signal_followup,
                mode="followup",
                primary_features=get_on_site_water_features(initial_analysis),
            )
            followup_ran = True

    analysis_after_followup = followup_analysis if followup_analysis else initial_analysis
    images_after_followup = followup_images if followup_images else initial_images

    remainder_rescan_ran = False
    remainder_rescan_analysis: Optional[Dict[str, Any]] = None

    if should_run_remainder_rescan(analysis_after_followup, polygon_present=bool(polygon)):
        primary_features = get_on_site_water_features(analysis_after_followup)

        remainder_candidate_images = [
            img for img in images_after_followup
            if img.get("type") in ("subzone", "historical_subzone", "historical_context")
            or str(img.get("label", "")).startswith("current_subzone_")
        ]

        if remainder_candidate_images:
            water_signal_remainder = simple_water_indicator(remainder_candidate_images)
            remainder_rescan_analysis = call_ai_screening(
                payload=payload,
                resolved=resolved,
                images=remainder_candidate_images,
                water_signal=water_signal_remainder,
                mode="remainder_rescan",
                primary_features=primary_features,
                max_images=min(12, MAX_AI_IMAGES),
            )
            remainder_rescan_ran = True

    final_analysis = analysis_after_followup
    final_images = images_after_followup
    image_lookup = {img.get("label"): img for img in final_images}

    if remainder_rescan_analysis:
        final_analysis = merge_analyses(
            analysis_after_followup,
            remainder_rescan_analysis,
            resolved=resolved,
            image_lookup=image_lookup,
        )
    else:
        final_analysis = sanitize_analysis_for_report(
            final_analysis,
            resolved=resolved,
            image_lookup=image_lookup,
        )

    hunter_mode_ran = False
    hunter_analysis: Optional[Dict[str, Any]] = None
    hunter_candidate_images: List[Dict[str, Any]] = []
    primary_features_for_hunter = get_on_site_water_features(final_analysis)
    established_former_truth = analysis_has_on_site_former_water_evidence(final_analysis)

    if polygon and primary_features_for_hunter:
        hunter_candidate_images = pick_hunter_images(final_images, primary_features_for_hunter, max_images=min(12, MAX_AI_IMAGES))
        if hunter_candidate_images:
            water_signal_hunter = simple_water_indicator(hunter_candidate_images)
            hunter_raw = call_ai_screening(
                payload=payload,
                resolved=resolved,
                images=hunter_candidate_images,
                water_signal=water_signal_hunter,
                mode="hunter",
                primary_features=primary_features_for_hunter,
                max_images=min(12, MAX_AI_IMAGES),
            )
            hunter_candidates = [
                f for f in safe_list(hunter_raw.get("distinct_features"))
                if isinstance(f, dict) and hunter_keep_feature(f, primary_features_for_hunter)
            ]
            if hunter_candidates:
                hunter_raw["distinct_features"] = hunter_candidates
                hunter_analysis = sanitize_analysis_for_report(
                    hunter_raw,
                    resolved=resolved,
                    image_lookup=image_lookup,
                )
                if established_former_truth and not analysis_has_on_site_former_water_evidence(hunter_analysis):
                    hunter_analysis = None
                else:
                    final_analysis = merge_analyses(
                        final_analysis,
                        hunter_analysis,
                        resolved=resolved,
                        image_lookup=image_lookup,
                    )
                    hunter_mode_ran = True

    # Final truth-layer sanitize and ID repair must run exactly once after all merge steps.
    final_analysis = sanitize_analysis_for_report(
        final_analysis,
        resolved=resolved,
        image_lookup=image_lookup,
    )

    # Hard carry-through rule: if any pass found a distinct historical-only on-site former pond,
    # do not allow later simplification to discard it.
    final_analysis = inject_force_promoted_former_ponds(
        final_analysis,
        analyses=[initial_analysis, followup_analysis, remainder_rescan_analysis, hunter_analysis],
        resolved=resolved,
        image_lookup=image_lookup,
    )
    final_analysis = sanitize_analysis_for_report(
        final_analysis,
        resolved=resolved,
        image_lookup=image_lookup,
    )
    final_analysis = enforce_former_water_truth_lock(
        final_analysis,
        analyses=[initial_analysis, followup_analysis, remainder_rescan_analysis, hunter_analysis],
    )

    report_generated = False
    report_filename = None
    report_url = None
    report_error = None

    try:
        report_filename = build_report_pdf(
            payload=payload,
            resolved=resolved,
            images=final_images,
            analysis=final_analysis,
        )
        report_generated = True
        report_url = str(request.base_url).rstrip("/") + f"/reports/{report_filename}"
    except Exception as exc:
        report_error = str(exc)

    return {
        "success": True,
        "address_input": payload.address,
        "location_used": {
            "lat": lat,
            "lng": lng,
            "source": resolved["location_source"],
            "matched_address": resolved.get("matched_address"),
        },
        "bbox_used": bbox,
        "context_bbox_used": context_bbox,
        "wide_context_bbox_used": wide_context_bbox,
        "subzones_used": subzones,
        "bbox_width_m": resolved.get("bbox_width_m"),
        "bbox_height_m": resolved.get("bbox_height_m"),
        "lot_area_m2": polygon_area_m2(polygon),
        "context_bbox_width_m": resolved.get("context_bbox_width_m"),
        "context_bbox_height_m": resolved.get("context_bbox_height_m"),
        "wide_context_bbox_width_m": resolved.get("wide_context_bbox_width_m"),
        "wide_context_bbox_height_m": resolved.get("wide_context_bbox_height_m"),
        "polygon_used": polygon,
        "automated_indicators": {
            "initial_water_signal": water_signal_initial,
        },
        "followup": {
            "ran": followup_ran,
            "years_added": followup_years,
            "reason": (
                "Initial pass flagged potential ponds/dams, fill/disturbance, or other visible geotechnical risk indicators, or chronology uncertainty."
                if followup_ran
                else "No historical follow-up deep scan triggered."
            ),
            "initial_analysis": initial_analysis,
            "followup_analysis": followup_analysis,
        },
        "remainder_rescan": {
            "ran": remainder_rescan_ran,
            "reason": (
                "Remainder-lot rescan triggered because the buried / former secondary pond risk was more geotechnically important than the already-obvious current pond."
                if remainder_rescan_ran
                else "No remainder-lot rescan triggered."
            ),
            "analysis": remainder_rescan_analysis,
        },
        "hunter_mode": {
            "ran": hunter_mode_ran,
            "reason": (
                "Low-confidence hunter mode ran to search for subtle secondary former ponds or buried pond footprints not carried through the main clean scan."
                if hunter_mode_ran
                else "Hunter mode did not return any new on-site secondary pond candidates."
            ),
            "images_used": [img.get("label") for img in hunter_candidate_images],
            "analysis": hunter_analysis,
        },
        "images": final_images,
        "analysis": final_analysis,
        "report_generated": report_generated,
        "report_filename": report_filename,
        "report_url": report_url,
        "report_error": report_error,
    }