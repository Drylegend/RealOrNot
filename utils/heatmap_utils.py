import spacy
import requests
import folium

# Load spaCy model once
_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
       try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_locations(text: str) -> list:
    """
    Use spaCy NER to extract unique GPE (geopolitical) entities from the text.
    Returns a list of location strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ == "GPE"})

def geocode_location(location: str, api_key: str) -> tuple:
    """
    Geocode a single location string via OpenCage API.
    Returns (lat, lng) or (None, None) if not found.
    """
    url = f"https://api.opencagedata.com/geocode/v1/json"
    params = {"q": location, "key": api_key}
    res = requests.get(url, params=params).json()
    if res.get("results"):
        geom = res["results"][0]["geometry"]
        return geom["lat"], geom["lng"]
    return None, None

def build_heatmap(text: str, api_key: str, start_zoom: int = 2, start_loc: tuple = (20, 0)) -> tuple:
    """
    Given input text and OpenCage API key, extract locations, geocode them,
    and build a Folium map with markers.
    Returns (folium.Map, list_of_locations).
    """
    locations = extract_locations(text)
    m = folium.Map(location=start_loc, zoom_start=start_zoom)
    for loc in locations:
        lat, lng = geocode_location(loc, api_key)
        if lat is not None and lng is not None:
            folium.CircleMarker(
                location=[lat, lng],
                radius=5,
                popup=loc,
                color="red"
            ).add_to(m)
    return m, locations
