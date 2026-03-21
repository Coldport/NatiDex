import json
import os
import re
import threading
import requests
from html.parser import HTMLParser
from pyinaturalist import get_observation_species_counts, get_observations


class _TextExtractor(HTMLParser):
    """Strips HTML tags, skips script/style/sup content, collects visible text."""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip  = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "sup"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "sup"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts)


def _extract_facts(text: str, title: str = "") -> dict:
    """Parse safety flags, size, and mass from plain Wikipedia intro text."""
    t = text.lower()
    n = title.lower()   # species/common name used for name-based hints

    # ── Venomous ──────────────────────────────────────────────────────
    # Text patterns: direct mentions of venom delivery
    venomous = bool(re.search(
        r'\bvenomous\b|\benvenomat\w+|\bvenom\b'
        r'|\bpit viper\b|\bviper\b|\badder\b|\bcobra\b|\bmamba\b'
        r'|\brattlesnake\b|\bkrait\b|\bboomslang\b|\bcopperhead\b'
        r'|\bfangs?\b|\bsting\b|\bstings\b|\bstinger\b'
        r'|\bnematocyst\b|\bspinnerets\b|\bparalyses? (?:its )?prey\b',
        t
    ))
    # Name-based: common names AND Latin genera for venomous insects/animals
    venomous = venomous or bool(re.search(
        # Common name hints
        r'\bbee\b|\bwasp\b|\bhornet\b|\bscorpion\b|\byellow.?jacket\b'
        r'|\bfire ant\b|\bjellyfish\b|\bsea anemone\b'
        # Latin genera: bees, wasps, hornets, ants
        r'|\bbombus\b|\bapis\b|\bvespula\b|\bvespa\b|\bpolistes\b'
        r'|\bsolenopsis\b|\bponera\b',
        n
    ))
    if re.search(r'\bnon.?venomous\b|\bnot venomous\b|\blacks? venom\b', t):
        venomous = False

    # ── Poisonous ─────────────────────────────────────────────────────
    poisonous = bool(re.search(
        r'\bpoisonous\b|\btoxic\b|\btoxin\b|\bpoison\b'
        r'|\bneurotoxin\b|\bhemotoxin\b|\bcardiotoxin\b|\btetrodotoxin\b'
        r'|\bpoisoning\b|\bintoxicat\w+\b'
        r'|\bunpalatable\b|\bdistasteful to\b'
        r'|\bdeter(?:s|red)? predators\b|\bchemical defen\w+\b',
        t
    ))
    if re.search(r'\bnot (?:poisonous|toxic)\b|\bnon.?(?:poisonous|toxic)\b', t):
        poisonous = False

    # ── Dangerous ─────────────────────────────────────────────────────
    dangerous = bool(re.search(
        r'\bdangerous\b|\bdeadly\b|\bfatal\b|\blethal\b'
        r'|\battacks?\s+humans?\b|\battacks?\s+people\b|\battacks?\s+livestock\b'
        r'|\bman.eating\b|\bman.?killer\b|\bhuman fatalities?\b'
        r'|\bkills?\s+humans?\b|\bkills?\s+people\b'
        r'|\bapex predator\b|\btop predator\b|\blarge predator\b'
        r'|\bknown to (?:attack|kill|bite)\b'
        r'|\bcan (?:kill|be fatal|be lethal)\b'
        r'|\bposes? a (?:danger|threat|risk) to humans?\b'
        r'|\bresponsible for (?:deaths?|fatalities?)\b'
        r'|\bferocious\b|\baggressive (?:toward|towards|to) humans?\b'
        # Taxonomic group mentions that imply a predatory mammal
        r'|\bspecies of canine\b|\bspecies of (?:bear|wolf|fox)\b'
        r'|\blarge (?:cat|feline|felid)\b|\bbig cat\b'
        r'|\bspecies of (?:crocodil|alligator)\b',
        t
    ))
    # Name-based: well-known dangerous animals whose Wikipedia intros
    # often avoid explicit "dangerous" language
    dangerous = dangerous or bool(re.search(
        r'\bcoyote\b|\bwolf\b|\bwolves\b|\bwolverine\b'
        r'|\blion\b|\btiger\b|\bleopard\b|\bjaguar\b'
        r'|\bcougar\b|\bpuma\b|\bmountain lion\b|\bcheetah\b|\blynx\b|\bbobcat\b'
        r'|\bgrizzly\b|\bpolar bear\b|\bbrown bear\b|\bblack bear\b|\bbear\b'
        r'|\bcrocodile\b|\bcrocodilian\b|\balligator\b|\bkomodo dragon\b'
        r'|\bgreat white\b|\bbull shark\b|\btiger shark\b|\bhammerhead\b'
        r'|\bmoose\b|\bwild boar\b|\bboar\b|\bcassowary\b'
        r'|\bhippopotamus\b|\bhippo\b|\bgila monster\b',
        n   # checked against title + species name
    ))

    def _find(keywords, units):
        kw = '|'.join(keywords)
        u  = '|'.join(units)
        # range: "grow to 10–35 cm"
        m = re.search(rf'(?:{kw}).{{0,25}}?(\d+(?:[.,]\d+)?)\s*(?:to|–|-)\s*(\d+(?:[.,]\d+)?)\s*({u})', t)
        if m:
            return f"{m.group(1)}\u2013{m.group(2)} {m.group(3)}"
        # single: "up to 30 cm"
        m = re.search(rf'(?:{kw}).{{0,20}}?(\d+(?:[.,]\d+)?)\s*({u})', t)
        return f"{m.group(1)} {m.group(2)}" if m else None

    size = _find(
        [r"reach", r"grow", r"length", r"long\b", r"height", r"tall\b", r"wingspan", r"up to"],
        [r"mm", r"cm", r"m\b", r"in\b", r"inch", r"feet", r"ft\b"],
    )
    mass = _find(
        [r"weigh", r"weight", r"mass"],
        [r"kg\b", r"g\b", r"lb\b", r"lbs\b", r"pound", r"oz\b"],
    )

    # ── Carnivore ─────────────────────────────────────────────────────
    carnivore = bool(re.search(
        r'\bcarnivore\b|\bcarnivorous\b'
        r'|\binsectivore\b|\binsectivorous\b'
        r'|\bpiscivore\b|\bpiscivorous\b'
        r'|\bpreys? on\b'
        r'|\bfeeds? (?:mainly |primarily |largely |exclusively )?on '
            r'(?:insects?|mammals?|birds?|fish|prey|worms?|invertebrates?|flesh|meat|rodents?)\b'
        r'|\beats? (?:mainly |primarily |largely )?'
            r'(?:insects?|mammals?|birds?|fish|prey|worms?|invertebrates?|flesh|meat|rodents?)\b'
        r'|\bdiet (?:consist|compris|made up).{0,25}'
            r'(?:insects?|mammals?|birds?|fish|meat|prey|flesh|worms?|invertebrates?)\b'
        r'|\bprimarily carnivorous\b|\bmainly carnivorous\b|\blargely carnivorous\b',
        t
    ))

    # ── Predator ──────────────────────────────────────────────────────
    predator = bool(re.search(
        r'\bpredatory\b|\bactive predator\b|\bambush predator\b'
        r'|\bapex predator\b|\btop predator\b|\blarge predator\b'
        r'|\bsit.and.wait predator\b|\bgeneralist predator\b|\bspecialist predator\b'
        r'|\bactive hunter\b|\bhunts? in packs?\b|\bhunts? cooperatively\b'
        r'|\bstalks? (?:its )?prey\b|\blies? in wait\b'
        r'|\bpursues? prey\b|\bchases? (?:its )?prey\b'
        r'|\bis an? (?:\w+ )?predator\b',
        t
    ))

    return {"venomous": venomous, "poisonous": poisonous, "dangerous": dangerous,
            "carnivore": carnivore, "predator": predator,
            "size": size, "mass": mass}


def _fetch_wiki(name: str, wiki_dir: str) -> None:
    """Fetch Wikipedia intro + thumbnail for a species; save to wiki_dir.

    name  – underscore form, e.g. 'Homo_sapiens'
    Files written:
      wiki/{name}.json    – title, description, extract_html, has_image, facts
      wiki/img/{name}.jpg – thumbnail (if available)
    Skipped if json already exists.
    """
    img_dir   = os.path.join(wiki_dir, "img")
    json_file = os.path.join(wiki_dir, f"{name}.json")
    img_file  = os.path.join(img_dir,  f"{name}.jpg")

    if os.path.exists(json_file):
        return

    os.makedirs(img_dir, exist_ok=True)

    latin = name.replace("_", " ")
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":      "query",
                "titles":      latin,
                "prop":        "extracts|pageimages",
                "exintro":     "1",
                "piprop":      "thumbnail",
                "pithumbsize": "500",
                "format":      "json",
                "redirects":   "1",
            },
            timeout=10,
            headers={"User-Agent": "NatiDex/1.0 (species identifier)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return
    page = next(iter(pages.values()))
    if "missing" in page:
        return

    extract_html = page.get("extract", "")
    thumb_url    = (page.get("thumbnail") or {}).get("source", "")
    has_image    = bool(thumb_url) and not os.path.exists(img_file)

    parser = _TextExtractor()
    parser.feed(extract_html)
    plain = parser.get_text()

    first_sentence = (plain.split(".")[0] + ".").strip() if plain else ""

    payload = {
        "title":        page.get("title", latin),
        "description":  first_sentence,
        "extract_html": extract_html,
        "has_image":    has_image or os.path.exists(img_file),
        "facts":        _extract_facts(plain, title=f"{page.get('title', latin)} {name}"),
    }

    if has_image:
        try:
            img_bytes = requests.get(thumb_url, timeout=8,
                                     headers={"User-Agent": "NatiDex/1.0"}).content
            with open(img_file, "wb") as f:
                f.write(img_bytes)
        except Exception:
            payload["has_image"] = False

    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError:
        pass


class DownloadController:
    def __init__(self):
        self._stop = threading.Event()
        self._running = False

    def is_running(self):
        return self._running

    def stop(self):
        self._stop.set()

    def start(self, on_update, species_limit=400, photos_per_species=50, data_dir="data"):
        self._running = True
        self._stop.clear()
        try:
            _download(on_update, self._stop, species_limit, photos_per_species, data_dir)
        finally:
            self._running = False


def _download(on_update, stop_event, species_limit, photos_per_species, data_dir):
    kingdoms = [1, 47126]  # Animals, Plants
    species_per_kingdom = species_limit // len(kingdoms)

    # ── Step 1: Gather all species metadata (paginated, API max 500/page) ──
    API_MAX = 500
    all_species = []
    for k_id in kingdoms:
        if stop_event.is_set():
            break
        on_update({"type": "download_status", "status": "fetching_species", "kingdom_id": k_id})
        page = 1
        while len(all_species) < species_limit:
            need = species_per_kingdom - sum(1 for s in all_species)
            counts = get_observation_species_counts(
                taxon_id=k_id, quality_grade='research',
                per_page=min(API_MAX, need), page=page
            )
            results = counts.get('results', [])
            if not results:
                break
            for record in results:
                taxon = record['taxon']
                all_species.append({
                    "name":         taxon['name'].replace(" ", "_"),
                    "display_name": taxon['name'],
                    "id":           taxon['id'],
                    "common_name":  taxon.get('preferred_common_name', ''),
                })
            if len(results) < min(API_MAX, need):
                break  # iNaturalist has no more results for this kingdom
            page += 1

    total_species = len(all_species)
    total_target = total_species * photos_per_species

    # Save common names mapping: {"Homo_sapiens": "Human", ...}
    common_names = {s["name"]: s["common_name"] for s in all_species if s["common_name"]}
    with open("common_names.json", "w", encoding="utf-8") as f:
        json.dump(common_names, f, ensure_ascii=False)

    on_update({
        "type": "download_init",
        "total_species": total_species,
        "species": [s["display_name"] for s in all_species],
        "photos_per_species": photos_per_species,
    })

    # ── Step 2: Download photos per species ─────────────────────────
    overall_downloaded = 0

    for idx, species in enumerate(all_species):
        if stop_event.is_set():
            break

        name = species["name"]
        display_name = species["display_name"]
        taxon_id = species["id"]
        save_path = os.path.join(data_dir, name)

        existing = len(os.listdir(save_path)) if os.path.exists(save_path) else 0

        # Fetch Wikipedia data (skipped if files already exist)
        _fetch_wiki(name, "wiki")

        # Skip species that already have enough photos
        if existing >= photos_per_species:
            overall_downloaded += photos_per_species
            on_update({
                "type": "download_species_done",
                "species": display_name,
                "species_idx": idx,
                "downloaded": photos_per_species,
                "total": photos_per_species,
                "skipped": True,
                "overall_pct": round(overall_downloaded / max(total_target, 1) * 100, 1),
                "overall_downloaded": overall_downloaded,
                "overall_total": total_target,
            })
            continue

        os.makedirs(save_path, exist_ok=True)
        on_update({
            "type": "download_species_start",
            "species": display_name,
            "species_idx": idx,
            "total_species": total_species,
        })

        # Paginate observations — API max 200/page; request extra since not
        # every observation has its photos field populated
        OBS_PAGE = 200
        downloaded = existing
        obs_page = 1
        while downloaded < photos_per_species and not stop_event.is_set():
            obs = get_observations(
                taxon_id=taxon_id, photos=True, quality_grade='research',
                per_page=OBS_PAGE, page=obs_page
            )
            results = obs.get('results', [])
            if not results:
                break
            for res in results:
                for photo in res.get('photos', []):
                    if stop_event.is_set() or downloaded >= photos_per_species:
                        break
                    img_url = photo['url'].replace('square', 'medium')
                    try:
                        img_data = requests.get(img_url, timeout=5).content
                        with open(os.path.join(save_path, f"{photo['id']}.jpg"), 'wb') as f:
                            f.write(img_data)
                        downloaded += 1
                        overall_downloaded += 1
                        on_update({
                            "type": "download_progress",
                            "species": display_name,
                            "species_idx": idx,
                            "downloaded": downloaded,
                            "total": photos_per_species,
                            "species_pct": round(downloaded / photos_per_species * 100, 1),
                            "overall_pct": round(overall_downloaded / max(total_target, 1) * 100, 1),
                            "overall_downloaded": overall_downloaded,
                            "overall_total": total_target,
                        })
                    except Exception:
                        continue
            if len(results) < OBS_PAGE:
                break  # no more observations available for this species
            obs_page += 1

        on_update({
            "type": "download_species_done",
            "species": display_name,
            "species_idx": idx,
            "downloaded": downloaded,
            "total": photos_per_species,
            "skipped": False,
            "overall_pct": round(overall_downloaded / max(total_target, 1) * 100, 1),
            "overall_downloaded": overall_downloaded,
            "overall_total": total_target,
        })

    if stop_event.is_set():
        on_update({"type": "download_status", "status": "stopped"})
    else:
        on_update({"type": "download_status", "status": "done"})
