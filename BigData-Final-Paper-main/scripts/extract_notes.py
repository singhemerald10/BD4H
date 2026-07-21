import os
import pandas as pd

def run():
    print("\nLoading data...")

    # Load raw CSVs
    notes_df = pd.read_csv("data/raw/NOTEEVENTS.csv")
    admits_df = pd.read_csv("data/raw/ADMISSIONS.csv")

    print(f"Total notes: {len(notes_df)}")
    print(f"Total admissions: {len(admits_df)}")

    # ✅ Filter discharge summaries only
    notes_df = notes_df[notes_df["CATEGORY"] == "Discharge summary"]
    print(f"Discharge summaries found: {len(notes_df)}")

    # ✅ Drop empty text or missing HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])
    print(f"After dropna (TEXT, HADM_ID): {len(notes_df)}")

    # Merge notes and admissions on HADM_ID
    merged_df = notes_df.merge(admits_df[["HADM_ID", "ADMITTIME", "DISCHTIME"]], on="HADM_ID", how="inner")

    # Sort by HADM_ID then ADMITTIME to compute next admissions
    merged_df["ADMITTIME"] = pd.to_datetime(merged_df["ADMITTIME"])
    merged_df["DISCHTIME"] = pd.to_datetime(merged_df["DISCHTIME"])
    admits_df["ADMITTIME"] = pd.to_datetime(admits_df["ADMITTIME"])
    admits_df["DISCHTIME"] = pd.to_datetime(admits_df["DISCHTIME"])
    # Create 30-day readmission label
    print("\nLabeling 30-day readmissions...")

    readmit_map = {}
    admits_df_sorted = admits_df.sort_values(by=["SUBJECT_ID", "ADMITTIME"])
    for i in range(len(admits_df_sorted) - 1):
        curr = admits_df_sorted.iloc[i]
        next_ = admits_df_sorted.iloc[i + 1]
        if curr["SUBJECT_ID"] == next_["SUBJECT_ID"]:
            days_between = (next_["ADMITTIME"] - curr["DISCHTIME"]).days
            if 0 <= days_between <= 30:
                readmit_map[curr["HADM_ID"]] = 1  # readmitted
            else:
                readmit_map[curr["HADM_ID"]] = 0  # not within 30
        else:
            readmit_map[curr["HADM_ID"]] = 0

    # Last admission per subject = no readmission
    last_hadm = admits_df_sorted.iloc[-1]["HADM_ID"]
    readmit_map[last_hadm] = 0

    merged_df["LABEL"] = merged_df["HADM_ID"].map(readmit_map)
    merged_df = merged_df.dropna(subset=["LABEL"])
    merged_df["LABEL"] = merged_df["LABEL"].astype(int)

    # Print stats
    label_counts = merged_df["LABEL"].value_counts().to_dict()
    print(f"Labeled admissions: {label_counts}")

    # Only keep latest discharge note per HADM_ID
    merged_df = merged_df.sort_values(by="DISCHTIME").drop_duplicates("HADM_ID", keep="last")

    print(f"🔗 Merged discharge notes with labels: {len(merged_df)}")
    print(f"   → Class balance: {merged_df['LABEL'].value_counts().to_dict()}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    merged_df[["HADM_ID", "TEXT", "LABEL"]].to_csv("data/processed/discharge_labeled.csv", index=False)
    print(f"Saved labeled discharge summaries: data/processed/discharge_labeled.csv ({len(merged_df)} rows)")
