from pydantic import BaseModel
from typing import List

class KandidatItem(BaseModel):
    id_barang: int
    nama_barang: str
    kategori: str
    deskripsi: str
    lokasi: str 
    foto_barang: str

class MatchPayload(BaseModel):
    id_target: int
    trigger_type: str
    nama_barang: str
    kategori: str
    deskripsi: str
    lokasi: str
    foto_barang: str
    
    candidates: List[KandidatItem]
    webhook_url: str