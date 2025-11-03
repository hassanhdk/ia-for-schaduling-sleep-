# ai_alarm_excel.py
"""
AI Alarm Scheduler (Excel version)
- Reads and writes sleep data from/to Excel (sleep_data.xlsx)
- Learns from your data using Machine Learning
- Predicts the next wake-up time automatically
"""

import os
import time
import datetime
import pickle
import pandas as pd
from dateutil import parser
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from playsound import playsound

# =============== SETTINGS ===============
EXCEL_FILE = "sleep_data.xlsx"
MODEL_FILE = "wake_model.pkl"
ALARM_SOUND = "alarm.wav"

NORMAL_SLEEP_HOURS = 7
HOLIDAY_SLEEP_HOURS = 9
HOLIDAYS = ["Saturday", "Sunday"]

# =============== HELPER FUNCTIONS ===============
def ensure_excel():
    """Create Excel file with headers if it doesn't exist."""
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=["date", "bed_time", "wake_time", "snooze_count", "is_holiday", "prev_sleep_minutes"])
        df.to_excel(EXCEL_FILE, index=False)
        print("📘 Created new Excel file:", EXCEL_FILE)

def read_excel_data():
    """Load data from Excel into a DataFrame."""
    if not os.path.exists(EXCEL_FILE):
        ensure_excel()
    return pd.read_excel(EXCEL_FILE)

def write_to_excel(record):
    """Append new sleep log to Excel file."""
    df = read_excel_data()
    new_row = pd.DataFrame([record])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    print("📝 Log added to Excel:", record)

def time_to_minutes(t):
    return t.hour * 60 + t.minute

def minutes_to_time(mins):
    mins = int(mins) % (24 * 60)
    return datetime.time(mins // 60, mins % 60)

def is_today_holiday():
    weekday = datetime.datetime.today().strftime("%A")
    return weekday in HOLIDAYS

# =============== MODEL TRAINING ===============
def prepare_dataframe(df):
    """Convert Excel data to numeric training set."""
    if df.empty:
        return df

    rows = []
    for _, r in df.iterrows():
        try:
            date = str(r["date"])
            bed = parser.parse(date + " " + str(r["bed_time"]))
            wake = parser.parse(date + " " + str(r["wake_time"]))
            if wake <= bed:
                wake += datetime.timedelta(days=1)
            sleep_minutes = (wake - bed).total_seconds() / 60.0
            rows.append({
                "bed_minutes": time_to_minutes(bed.time()),
                "weekday": bed.weekday(),
                "is_holiday": int(r.get("is_holiday", 0)),
                "snooze_count": int(r.get("snooze_count", 0)),
                "prev_sleep_minutes": float(r.get("prev_sleep_minutes", sleep_minutes)),
                "wake_minutes": time_to_minutes(wake.time())
            })
        except Exception as e:
            print("⚠️ Skipped invalid row:", r.to_dict(), e)
    return pd.DataFrame(rows)

def train_model(df):
    if df.shape[0] < 5:
        print("⚠️ Not enough Excel data (<5 rows). Using default rules.")
        return None

    X = df[["bed_minutes", "weekday", "is_holiday", "snooze_count", "prev_sleep_minutes"]]
    y = df["wake_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ Model trained (MAE: {mae:.1f} minutes)")
    return model

def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

# =============== AI LOGIC ===============
def predict_wake_time(model, bed_time, is_holiday, prev_sleep):
    if model is None:
        base = HOLIDAY_SLEEP_HOURS if is_holiday else NORMAL_SLEEP_HOURS
        wake_dt = (datetime.datetime.combine(datetime.date.today(), bed_time) +
                   datetime.timedelta(hours=base))
        return wake_dt.time()

    features = pd.DataFrame([{
        "bed_minutes": time_to_minutes(bed_time),
        "weekday": datetime.date.today().weekday(),
        "is_holiday": int(is_holiday),
        "snooze_count": 0,
        "prev_sleep_minutes": prev_sleep if prev_sleep else 60 * NORMAL_SLEEP_HOURS
    }])
    predicted_minutes = model.predict(features)[0]
    return minutes_to_time(predicted_minutes)

def ring_alarm():
    print("\n⏰ Wake up! Time to start your day!")
    if os.path.exists(ALARM_SOUND):
        try:
            playsound(ALARM_SOUND)
        except Exception as e:
            print("Sound error:", e)
    else:
        for _ in range(5):
            print("BEEP! BEEP!")
            time.sleep(0.6)

# =============== MAIN LOOP ===============
def main():
    ensure_excel()
    df_raw = read_excel_data()
    df_train = prepare_dataframe(df_raw)
    model = train_model(df_train)
    if model:
        save_model(model)
    else:
        model = load_model()

    bed_time_str = input("Enter your bedtime (HH:MM): ") or "00:30"
    bed_time = datetime.datetime.strptime(bed_time_str, "%H:%M").time()
    is_hol = is_today_holiday()

    prev_sleep = df_train["sleep_minutes"].iloc[-1] if "sleep_minutes" in df_train and not df_train.empty else None
    wake_time = predict_wake_time(model, bed_time, is_hol, prev_sleep)

    print(f"🕒 Bedtime: {bed_time_str} | Holiday: {is_hol}")
    print(f"🔮 Predicted wake time: {wake_time.strftime('%H:%M')}")

    while True:
        now = datetime.datetime.now().time()
        if now.hour == wake_time.hour and now.minute == wake_time.minute:
            ring_alarm()
            bed_dt = datetime.datetime.combine(datetime.date.today(), bed_time)
            if datetime.datetime.now() <= bed_dt:
                bed_dt -= datetime.timedelta(days=1)
            sleep_minutes = (datetime.datetime.now() - bed_dt).total_seconds() / 60.0

            record = {
                "date": datetime.date.today().isoformat(),
                "bed_time": bed_time_str,
                "wake_time": wake_time.strftime("%H:%M"),
                "snooze_count": 0,
                "is_holiday": is_hol,
                "prev_sleep_minutes": sleep_minutes
            }
            write_to_excel(record)
            break
        time.sleep(15)

if __name__ == "__main__":
    main()
