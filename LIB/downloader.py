import os
import requests
from pyinaturalist import get_observation_species_counts, get_observations

def download_species_dataset(total_target=20000, species_limit=400, photos_per_species=50):
    """
    Downloads the top observed species to ensure the model has enough data per class.
    """
    data_dir = "data"
    # Kingdom IDs: 1 (Animals), 47126 (Plants)
    kingdoms = [1, 47126]
    species_per_kingdom = species_limit // len(kingdoms)

    for k_id in kingdoms:
        print(f"\n--- Finding Top {species_per_kingdom} species for Kingdom ID: {k_id} ---")
        
        # 1. Get the most frequently observed species
        counts = get_observation_species_counts(taxon_id=k_id, quality_grade='research', per_page=species_per_kingdom)
        
        for record in counts['results']:
            taxon = record['taxon']
            name = taxon['name'].replace(" ", "_")
            taxon_id = taxon['id']
            
            save_path = os.path.join(data_dir, name)
            if os.path.exists(save_path) and len(os.listdir(save_path)) >= photos_per_species:
                continue
            
            os.makedirs(save_path, exist_ok=True)
            print(f"Downloading {photos_per_species} photos for: {name}")

            # 2. Get the photos for this specific species
            obs = get_observations(taxon_id=taxon_id, photos=True, quality_grade='research', per_page=photos_per_species)
            
            downloaded = 0
            for res in obs['results']:
                for photo in res.get('photos', []):
                    if downloaded >= photos_per_species: break
                    
                    img_url = photo['url'].replace('square', 'medium')
                    try:
                        img_data = requests.get(img_url, timeout=5).content
                        with open(os.path.join(save_path, f"{photo['id']}.jpg"), 'wb') as f:
                            f.write(img_data)
                        downloaded += 1
                    except: continue