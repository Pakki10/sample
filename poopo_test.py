import os
import json
import pandas as pd
import requests
from typing import List, Dict, Any
from tqdm import tqdm
import time

def load_hsn_data(excel_path: str) -> Dict[str, str]:
    """
    Load HSN codes and descriptions from Excel file.
    Returns a dictionary mapping HSN_CD to HSN_Description.
    Ignores any descriptions that contain 'other' in any case variation.
    """
    try:
        df = pd.read_excel(excel_path)
        hsn_mapping = {}
        for _, row in df.iterrows():
            hsn_cd = str(row['HSN_CD']).strip()
            hsn_desc = str(row['HSN_Description']).strip()
            if not hsn_cd or not hsn_desc or hsn_cd == 'nan' or hsn_desc == 'nan':
                continue
            if hsn_desc.lower() == 'other':
                continue
            hsn_mapping[hsn_cd] = hsn_desc
        return hsn_mapping
    except Exception as e:
        print(f"Error loading HSN data: {e}")
        return {}

def normalize_hsn_code(hsn_code: str) -> str:
    """
    Normalize HSN code by removing spaces and converting to string.
    """
    return str(hsn_code).replace(" ", "").strip()

def check_hsn_reference(text: str, hsn_description: str) -> str:
    """
    Use local LLM to check if HSN description is mentioned in the text.
    Returns 'Yes' if mentioned, 'No' if not, 'Error' if API call fails.
    """
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "meta-llama-3.1-8b-instruct",
        "messages": [
            {
                "role": "system",
                "content": "Strictly answer only Yes or No. Do not explain."
            },
            {
                "role": "user",
                "content": f"Is the following HSN description mentioned in the text?\n\nText:\n{text}\n\nHSN Description:\n{hsn_description}"
            }
        ],
        "temperature": 0,
        "max_tokens": -1,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.ok:
            reply = response.json()['choices'][0]['message']['content'].strip()
            return "Yes" if "yes" in reply.lower() else "No"
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return "Error"
    except Exception as e:
        print(f"Exception calling LLM API: {e}")
        return "Error"

def find_hsn_matches_in_notification(notification: Dict[str, Any], hsn_mapping: Dict[str, str]) -> List[str]:
    """
    Find HSN descriptions that match in the notification text or name using LLM.
    Returns list of HSN codes that match.
    """
    matches = []
    search_text = f"{notification.get('text', '')} {notification.get('notificationName', '')}".strip()
    if not search_text:
        print(f"Empty text for notification {notification.get('id')}")
        return matches
    for hsn_code, hsn_description in hsn_mapping.items():
        clean_description = hsn_description.strip()
        if clean_description.lower() == 'other' or len(clean_description) < 5:
            continue
        llm_result = check_hsn_reference(search_text, clean_description)
        if llm_result == "Yes":
            normalized_hsn = normalize_hsn_code(hsn_code)
            matches.append(normalized_hsn)
    return matches

def update_notification_hsn_refs_only_empty(notifications: List[Dict[str, Any]], hsn_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Only update notifications where HSN_ref is missing or empty, performing LLM matching.
    Other notifications are kept unchanged.
    Report progress per-notification and overall, with elapsed time.
    """
    updated_notifications = []
    n_total = len(notifications)
    t0_total = time.time()

    # Count notifications to process (empty HSN_ref only)
    todo_indices = [i for i, n in enumerate(notifications) if not n.get('HSN_ref', [])]
    n_todo = len(todo_indices)

    print(f"Processing {n_todo} of {n_total} notifications (with empty/missing HSN_ref).")

    # Main loop with tqdm progress bar
    with tqdm(total=n_todo, desc="Notifications processed") as notif_bar:

        for i, notification in enumerate(notifications):
            if not notification.get('HSN_ref', []):
                t0 = time.time()
                # Progress bar for HSN loop, shown per notification
                matches = []
                try:
                    with tqdm(total=len(hsn_mapping), leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} HSNs [{elapsed}]', desc=f"Notif {i+1}/{n_total}") as hsn_bar:
                        search_text = f"{notification.get('text', '')} {notification.get('notificationName', '')}".strip()
                        for hsn_code, hsn_description in hsn_mapping.items():
                            clean_desc = hsn_description.strip()
                            if clean_desc.lower() == "other" or len(clean_desc) < 5:
                                hsn_bar.update(1)
                                continue
                            llm_result = check_hsn_reference(search_text, clean_desc)
                            if llm_result == "Yes":
                                matches.append(normalize_hsn_code(hsn_code))
                            hsn_bar.update(1)
                    elapsed = time.time() - t0
                except Exception as e:
                    print(f"Error processing notification {i}: {e}")
                    elapsed = time.time() - t0

                notification_new = notification.copy()
                notification_new["HSN_ref"] = matches
                updated_notifications.append(notification_new)
                notif_bar.set_postfix({
                    "current": i + 1,
                    "pending": n_todo - notif_bar.n - 1,
                    "last_time": f"{elapsed:.1f}s"
                })
                notif_bar.update(1)
            else:
                updated_notifications.append(notification)
        print(f"\nTOTAL elapsed time: {time.time() - t0_total:.1f}s")
    return updated_notifications

def process_notifications_with_hsn_matching_only_empty(json_path: str, excel_path: str, output_path: str = None):
    """
    Main function to process only notifications with missing/empty HSN_ref field.
    Shows progress bars and elapsed time.
    """
    try:
        print(f"\n🚀 STARTING HSN MATCHING PROCESS (ONLY for empty/missing HSN_ref)")
        hsn_mapping = load_hsn_data(excel_path)
        with open(json_path, "r", encoding="utf-8") as f:
            notifications = json.load(f)
        updated_notifications = update_notification_hsn_refs_only_empty(notifications, hsn_mapping)
        if output_path is None:
            output_path = json_path.replace('.json', '_updated.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_notifications, f, indent=2, ensure_ascii=False)
        print(f"✅ COMPLETED - Updated notifications saved to: {output_path}")
    except Exception as e:
        print(f"❌ ERROR: Process failed - {e}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(BASE_DIR, "notifications_final.json")
    EXCEL_PATH = os.path.join(BASE_DIR, "HSN_SAC_Enriched_N.xlsx")
    OUTPUT_PATH = os.path.join(BASE_DIR, "notifications_final_updated.json")
    process_notifications_with_hsn_matching_only_empty(JSON_PATH, EXCEL_PATH, OUTPUT_PATH)
