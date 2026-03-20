from LIB.downloader import download_species_dataset
from LIB.train import train_species_classifier

def main():
    # 1. Download: 200 animal species + 200 plant species (50 photos each)
    print("=== STARTING SPECIES DATASET COLLECTION ===")
    download_species_dataset(total_target=1000000, species_limit=5000, photos_per_species=100)

    # 2. Train: Automatically detects number of species based on folders
    print("\n=== STARTING MULTI-SPECIES TRAINING ===")
    train_species_classifier(data_dir="data")

if __name__ == "__main__":
    main()