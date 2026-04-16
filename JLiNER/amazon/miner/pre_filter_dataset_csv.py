import pandas as pd
import re


# Load your dataset
df = pd.read_csv('/Users/scottwang/PycharmProjects/JSD/JLiNER/reserve_data/test.csv')

def extract_balanced_dataset(df):
    # --- 1. REFINED REGEX PATTERNS ---

    # Exclude pets for the children category
    pet_exclusion = r'(dog|cat|puppy|pupper|kitten|pet|fur.baby|paw)'

    # Children: Look for age/relation but ensure it's not a pet
    # Uses negative lookahead to skip sentences mentioning pets nearby
    children_regex = rf'\b(son|daughter|nephew|niece|grandchild|toddler|infant|baby|kids?|child(ren)?|grade|school|pediatrician)\b'

    # Medical: Focus on clinical/symptomatic language
    medical_regex = r'\b(diagnosed|symptoms|chronic|prescription|dosage|treatment|recovery|allergy|disease|doctor|physician|medication|pain|ailment|illness|side.effects)\b'

    # Gender: Author identity markers and familial gender roles
    gender_regex = r'\b(husband|wife|boyfriend|girlfriend|gentleman|lady|as.a.(woman|man|girl|boy|female|male)|myself.as)\b'

    # --- 2. INITIALIZE BUCKETS ---
    buckets = {
        'children': [],
        'medical': [],
        'gender': [],
        'non_pii': []
    }

    limits = {
        'children': 1500,
        'medical': 1500,
        'gender': 1500,
        'non_pii': 0
    }

    # --- 3. PROCESSING LOOP ---
    # Shuffle the data first to ensure variety
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for _, row in df.iterrows():
        text = str(row['ori_review']).lower()
        assigned = False

        # Check if all limits are reached to break early
        if all(len(buckets[k]) >= limits[k] for k in limits):
            break

        # Check for Children (with pet exclusion)
        if len(buckets['children']) < limits['children']:
            if re.search(children_regex, text) and not re.search(pet_exclusion, text):
                row['hint_category'] = 'children'
                buckets['children'].append(row)
                assigned = True

        # Check for Medical
        if not assigned and len(buckets['medical']) < limits['medical']:
            if re.search(medical_regex, text):
                row['hint_category'] = 'medical'
                buckets['medical'].append(row)
                assigned = True

        # Check for Gender
        if not assigned and len(buckets['gender']) < limits['gender']:
            if re.search(gender_regex, text):
                row['hint_category'] = 'gender'
                buckets['gender'].append(row)
                assigned = True

        # If no matches, assign to Non-PII
        if not assigned and len(buckets['non_pii']) < limits['non_pii']:
            row['hint_category'] = 'non-pii'
            buckets['non_pii'].append(row)

    # Combine all buckets into one DataFrame
    final_list = buckets['children'] + buckets['medical'] + buckets['gender'] + buckets['non_pii']
    result_df = pd.DataFrame(final_list)

    return result_df

# To use:
filtered_data = extract_balanced_dataset(df)
filtered_data.to_csv('balanced_annotation_task.csv', index=False)