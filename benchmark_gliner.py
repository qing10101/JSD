import pandas as pd
import time
from gliner import GLiNER

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
GLINER_MODEL_NAME = "numind/NuNER_Zero-span"
DEFAULT_THRESHOLD = 0.3


# ------------------------------------------------------------------
# 1. LOAD MODEL (Global Resource)
# ------------------------------------------------------------------
print(f"⏳ Loading GLiNER model: {GLINER_MODEL_NAME}...")
try:
    # Load once to be used by the pipeline function
    gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading GLiNER: {e}")
    exit()


# ------------------------------------------------------------------
# 2. PIPELINE FUNCTION
# ------------------------------------------------------------------
def extract_pii_from_profile(profile_data, labels=None, threshold=DEFAULT_THRESHOLD):
    """
    Pipeline Entry Point for PII Extraction.

    Args:
        profile_data (dict): Profile object (e.g. {"id": "1", "comments": ["text", ...]}).
        labels (list): List of entity strings to search for. Defaults to DEFAULT_LABELS.
        threshold (float): Confidence cutoff.

    Returns:
        list: A list of dictionaries containing detected entities.
    """
    if labels is None:
        print('No label provided.')
        return []

    # 1. Prepare Text (Join all comments into one block)
    # This matches the logic in the LLM inference script
    comments_list = profile_data.get("comments", [])
    if isinstance(comments_list, str):
        comments_list = [comments_list]

    if not comments_list:
        return []

    profile_id = profile_data.get("id", "Unknown")
    raw_text = "\n".join(comments_list)

    # 2. Run Inference
    start_time = time.time()
    try:
        entities = gliner_model.predict_entities(
            raw_text,
            labels,
            threshold=threshold,
            multi_label=True
        )
        duration = time.time() - start_time

        # 3. Format Results
        results = []
        for e in entities:
            results.append({
                "Profile ID": profile_id,
                "Entity Type": e['label'],
                "Text Detected": e['text'],
                "Confidence": round(e['score'], 2),
                "Time": round(duration, 2)
            })

        return results

    except Exception as e:
        print(f"Error processing Profile {profile_id}: {e}")
        return []


# ------------------------------------------------------------------
# EXAMPLE USAGE (Pipeline Test)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --- Simulate a profile object (Same format as LLM pipeline) ---
    sample_profile_variable = {
        "id": "Test_User_01",
        "comments": [
            "I have been working as a Radiologist for 10 years.",
            "My back hurts from long shifts at the hospital in Seattle.",
            "I love playing chess on weekends."
        ],
        # 'ground_truth' is ignored here, as this is an extraction task
    }

    print("\n--- Pipeline Started (GLiNER) ---")

    # Pass the variable directly
    pii_results = extract_pii_from_profile(
        profile_data=sample_profile_variable,
        labels=["location", "occupation", "age", 'sex', 'medical condition']
    )

    # Output results
    df = pd.DataFrame(pii_results)
    if not df.empty:
        print(df.to_markdown(index=False))
    else:
        print("No PII detected.")