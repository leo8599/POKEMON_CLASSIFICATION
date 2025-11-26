import random, os, shutil

train_dir = "data/train"
test_dir = "data/test"

def prep_test_data(pokemon, train_dir, test_dir):
    pop = os.listdir(os.path.join(train_dir, pokemon))
    # Safety check: take 15 or however many exist if less than 15
    sample_size = min(len(pop), 15)
    test_data = random.sample(pop, sample_size)
    
    for f in test_data:
        # Source path
        src_path = os.path.join(train_dir, pokemon, f)
        # Destination path
        dst_path = os.path.join(test_dir, pokemon, f)
        shutil.copy(src_path, dst_path)

# Create the main test directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

for poke in os.listdir(train_dir):
    # Skip hidden files
    if poke.startswith("."):
        continue
    
    # Create the specific pokemon folder in the CORRECT location
    os.makedirs(os.path.join(test_dir, poke), exist_ok=True)
    
    # Copy the data
    prep_test_data(poke, train_dir, test_dir)
        
print('Test folder generation complete!')