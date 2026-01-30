filename = '/developing/yelp_dataset/yelp_academic_dataset_review.json'
count = 0

print(f"Counting entries in {filename}...")

try:
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
    print(f"Total reviews: {count}")
except FileNotFoundError:
    print(f"File {filename} not found. Please check the path.")