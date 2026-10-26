# Runs BEFORE any coaching. Medical-sounding input gets a fixed message and stops.
DISCLAIMER = ("General wellbeing information only, not medical advice. "
              "If you have serious or persistent symptoms, see a doctor.")

RED_FLAGS = {
    "self_harm": ["suicid", "kill myself", "end my life", "self-harm", "self harm",
                  "hurt myself", "don't want to wake up", "dont want to wake up"],
    "breathing_or_chest": ["chest pain", "can't breathe", "cannot breathe",
                           "stop breathing", "stops breathing", "gasping",
                           "choking in my sleep", "sleep apnea", "sleep apnoea"],
    "severe_symptoms": ["hallucinat", "seizure", "fainted", "passed out",
                        "haven't slept in days", "havent slept in days",
                        "not slept in days", "no sleep for days"],
    "medication": ["sleeping pill", "overdose", "melatonin dose", "zolpidem",
                   "zopiclone", "diazepam"],
}
MESSAGES = {
    "self_harm": ("I'm not able to help with this as a sleep coach, but you deserve "
                  "support right now. If you may be in danger, contact your local "
                  "emergency number or a crisis line, or reach out to someone you trust."),
    "breathing_or_chest": ("Symptoms like these need a medical professional. Please "
                           "speak to a doctor soon, or use emergency services if "
                           "they are severe or sudden."),
    "severe_symptoms": ("This sounds beyond habit coaching. Please see a doctor, "
                        "especially if it keeps happening or worries you."),
    "medication": ("I can't advise on medication or doses. Please ask a doctor or "
                   "pharmacist."),
}


def check_red_flags(text):
    t = text.lower().replace("\u2019", "'")
    for category, phrases in RED_FLAGS.items():
        for phrase in phrases:
            if phrase in t:
                return {"escalate": True, "category": category,
                        "message": MESSAGES[category]}
    return {"escalate": False, "category": None, "message": None}
