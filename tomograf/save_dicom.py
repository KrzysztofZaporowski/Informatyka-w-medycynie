import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import matplotlib.pyplot as plt
import numpy as np


def _normalizuj_date_dicom(data_badania):
    cyfry = ''.join(ch for ch in data_badania if ch.isdigit())
    if len(cyfry) >= 8:
        return cyfry[:8]
    raise ValueError('Data badania musi zawierac co najmniej 8 cyfr (YYYYMMDD lub YYYY-MM-DD).')

def zapisz_dicom(obraz, nazwa_pliku, dane_pacjenta, data_badania, komentarze):
    obraz = (obraz * 255).astype(np.uint8)

    imie, nazwisko, pesel = dane_pacjenta
    ds = Dataset()
    ds.PatientName = f"{imie} {nazwisko}"
    ds.PatientID = pesel
    ds.StudyDate = _normalizuj_date_dicom(data_badania)
    ds.StudyTime = '120000'
    ds.StudyDescription = komentarze
    ds.ImageComments = komentarze
    ds.Rows, ds.Columns = obraz.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    
    ds.PixelData = obraz.tobytes()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(nazwa_pliku, write_like_original=False)

    wczytany_dicom = pydicom.dcmread(nazwa_pliku)

    plt.imshow(wczytany_dicom.pixel_array, cmap='gray')
    plt.title(f"Zapisany DICOM\nPacjent: {wczytany_dicom.PatientName}, ID: {wczytany_dicom.PatientID}")
    plt.show()


def odczytaj_dicom(nazwa_pliku):
    ds = pydicom.dcmread(nazwa_pliku)
    obraz = ds.pixel_array.astype(np.float64)
    if np.max(obraz) > 0:
        obraz = obraz / np.max(obraz)

    meta = {
        'PatientName': str(getattr(ds, 'PatientName', '')),
        'PatientID': str(getattr(ds, 'PatientID', '')),
        'StudyDate': str(getattr(ds, 'StudyDate', '')),
        'StudyDescription': str(getattr(ds, 'StudyDescription', '')),
    }

    return obraz, meta