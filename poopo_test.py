import json
import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Set API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Model selection
MODEL = "gpt-4.1-nano"

# Load notifications JSON
with open("notifications.json", "r", encoding="utf-8") as f:
    notifications = json.load(f)

# Load HSN Excel
df_hsn = pd.read_excel("HSN_SAC_Enriched.xlsx", dtype=str)
hsn_dict = dict(zip(df_hsn["HSN_CD"].str.strip(), df_hsn["HSN_Description"].str.strip()))

# Prepare notification lookup
notif_lookup = {entry["notificationNo"]: entry["id"] for entry in notifications}
unique_notification_numbers = list(notif_lookup.keys())

# Helper: Find cross-referenced notifications
def find_referenced_notifications(text, current_notification, all_notification_nos):
    candidate_list = [n for n in all_notification_nos if n != current_notification]
    candidate_str = "\n".join(candidate_list[:100])

    prompt = f"""
The following is a customs notification. Identify all notification numbers (from the list) that are referenced within it.
Exclude the current notification itself: {current_notification}

Text:
{text}

Notification Numbers to Check Against:
{candidate_str}

Only return the list of matched notification numbers from the list above, one per line, no explanation.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        return [line for line in answer.splitlines() if line in notif_lookup]
    except Exception as e:
        print(f"⚠️ Reference check error: {e}")
        return []

# Helper: Find HSN references
def find_referenced_hsn(text, hsn_code_map):
    hsn_sample = "\n".join([f"{code}: {desc}" for code, desc in list(hsn_code_map.items())[:100]])
    prompt = f"""
You are given a customs notification text. Identify all HSN codes that are clearly mentioned, either directly or by matching the description.

Notification Text:
{text}

Available HSN Codes and Descriptions:
{hsn_sample}

Return only a list of matching HSN codes (not descriptions), one per line.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        return [line for line in answer.splitlines() if line in hsn_code_map]
    except Exception as e:
        print(f"⚠️ HSN detection error: {e}")
        return []

# Worker for threading
def process_notification(entry):
    current_notif_no = entry["notificationNo"]
    combined_text = f"{entry.get('notificationName', '')}\n{entry.get('text', '')}"

    referenced = find_referenced_notifications(combined_text, current_notif_no, unique_notification_numbers)
    entry["referencedNotifications"] = [{"notificationNo": n, "id": notif_lookup[n]} for n in referenced]

    matched_hsn = find_referenced_hsn(combined_text, hsn_dict)
    entry["matched_hsn"] = matched_hsn

    return entry

# Process in parallel (10 threads)
enriched = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_notification, entry) for entry in notifications]

    for future in tqdm(as_completed(futures), total=len(futures), desc="🔍 Enriching notifications"):
        enriched.append(future.result())

# Save output
with open("notifications_enriched.json", "w", encoding="utf-8") as f:
    json.dump(enriched, f, indent=2, ensure_ascii=False)

print("✅ Enrichment complete. File saved as notifications_enriched.json")
