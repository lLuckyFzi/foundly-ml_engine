from fastapi import APIRouter, BackgroundTasks
import requests
from io import BytesIO
from PIL import Image
import json

from schemas.payload import MatchPayload
from core.ml_engine import get_image_similarity, get_text_similarity, get_final_score, MINIMUM_SCORE

router = APIRouter()

def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error fetching image dari {url}: {e}")
        return None

def run_matching_engine(data: MatchPayload):
    try:
        img_target = load_image_from_url(data.foto_barang)
        if not img_target:
            requests.post(data.webhook_url, json={"status": "ERROR", "notes": "Gagal load gambar target."})
            return

        target_text_combined = f"{data.nama_barang}. Kategori: {data.kategori}. Lokasi: {data.lokasi}. Ciri-ciri: {data.deskripsi}"

        best_candidate_id = None
        best_score = 0.0

        for kandidat in data.candidates:
            print(f"--> Analisis Kandidat ID: {kandidat.id_barang}")
            
            if data.kategori.lower() != kandidat.kategori.lower():
                print("    [SKIPPED] Kategori berbeda.")
                continue

            img_kandidat = load_image_from_url(kandidat.foto_barang)
            if not img_kandidat:
                print("    [SKIPPED] Gagal load gambar kandidat.")
                continue

            kandidat_text_combined = f"{kandidat.nama_barang}. Kategori: {kandidat.kategori}. Lokasi: {kandidat.lokasi}. Ciri-ciri: {kandidat.deskripsi}"

            img_score = get_image_similarity(img_target, img_kandidat)
            text_score = get_text_similarity(target_text_combined, kandidat_text_combined)
            final_score = get_final_score(img_score, text_score)

            print(f"    [SKOR SEMENTARA] Kandidat {kandidat.id_barang} = {final_score:.2f}%")

            if final_score > best_score:
                best_score = final_score
                best_candidate_id = kandidat.id_barang

        status_kecocokan = False
        catatan = ""
        
        if best_candidate_id and best_score >= MINIMUM_SCORE:
            status_kecocokan = True
            catatan = "Memenuhi standar kecocokan Foundly."
        else:
            catatan = "Tidak ada kandidat yang memenuhi skor minimal 85%."

        payload_balasan = {
            "id_target": data.id_target,
            "id_kandidat_terbaik": best_candidate_id,
            "tingkat_kemiripan": round(best_score, 2),
            "status_kecocokan": status_kecocokan,
            "catatan": catatan
        }

        print("\n[MENGIRIM KE WEBHOOK NODE.JS]")
        print(json.dumps(payload_balasan, indent=4))
        
        requests.post(data.webhook_url, json=payload_balasan)

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        requests.post(data.webhook_url, json={"status": "ERROR", "notes": str(e)})


@router.post("/match")
def process_matching(payload: MatchPayload, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(run_matching_engine, payload)
    return {"message": f"Sistem menerima tugas 1-to-N. Memproses di background..."}